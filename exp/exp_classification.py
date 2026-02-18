from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
from torch import optim
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import time
import warnings
import numpy as np
import pdb
import shap
import matplotlib.pyplot as plt
import json

warnings.filterwarnings('ignore')

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
    
    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.pred_len = 0
        self.args.enc_in = len(train_data.ts_target_col)
        self.args.num_class = 4

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader, scaler = data_provider(self.args, flag, scaler=getattr(self, 'scaler', None))
        self.scaler = scaler  # Store scaler for reuse
        return data_set, data_loader

    def _select_optimizer(self):
        # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze())
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()

        
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label) in enumerate(tqdm(train_loader)):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(tqdm(test_loader)):
                batch_x = batch_x.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def xai_shap(self, setting, test_data, test_loader, num_samples=100):
        """
        SHAP 기반 XAI 분석 (KernelExplainer 사용)
        
        Args:
            setting: 모델 설정명
            test_data: 테스트 데이터
            test_loader: 테스트 데이터 로더
            num_samples: SHAP 분석에 사용할 샘플 수
        """
        print("\n" + "="*50)
        print("Starting SHAP-based XAI Analysis...")
        print("="*50)
        
        # 결과 저장 폴더
        xai_folder = './xai_results/' + setting + '/'
        if not os.path.exists(xai_folder):
            os.makedirs(xai_folder)
        
        self.model.eval()
        
        # 데이터 준비
        background_samples = []
        test_samples = []
        test_labels = []
        
        print("Preparing data for SHAP analysis...")
        for i, (batch_x, label) in enumerate(tqdm(test_loader)):
            if len(background_samples) < num_samples:
                background_samples.append(batch_x.numpy())
            if len(test_samples) < num_samples:
                test_samples.append(batch_x.numpy())
                test_labels.append(label.numpy())
            
            if len(background_samples) >= num_samples and len(test_samples) >= num_samples:
                break
        
        background_data = np.vstack(background_samples[:num_samples])
        X_test = np.vstack(test_samples[:num_samples])
        y_test = np.hstack(test_labels[:num_samples])
        
        print(f"Background data shape: {background_data.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        # 모델을 위한 wrapper 함수 - numpy 입력을 받아 numpy 출력
        def model_predict_wrapper(x):
            """
            모델 예측 함수
            Args:
                x: numpy array (batch_size, seq_len, features)
            Returns:
                probabilities: numpy array (batch_size, num_classes)
            """
            try:
                with torch.no_grad():
                    # numpy to tensor
                    x_tensor = torch.from_numpy(x).float().to(self.device)
                    
                    # 입력 shape 확인
                    if x_tensor.dim() == 2:
                        # (batch_size, features) shape인 경우 (seq_len, features)로 확장
                        batch_size = x_tensor.shape[0]
                        x_tensor = x_tensor.unsqueeze(1)  # (batch_size, 1, features)
                    
                    # 모델 forward pass
                    outputs = self.model(x_tensor)
                    
                    # softmax 적용
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    return probs.cpu().numpy()
            except Exception as e:
                print(f"Error in model_predict_wrapper: {e}")
                print(f"Input shape: {x.shape}")
                raise
        
        # 배경 데이터 평탄화 - SHAP이 1D/2D 입력을 선호
        background_data_flat = background_data.reshape(background_data.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        print(f"Flattened background data shape: {background_data_flat.shape}")
        print(f"Flattened test data shape: {X_test_flat.shape}")
        
        # 예측 함수 (1D 입력 처리)
        def predict_flat(x):
            """
            평탄화된 입력을 받는 예측 함수
            Args:
                x: numpy array (batch_size, flattened_features)
            Returns:
                probabilities: numpy array (batch_size, num_classes)
            """
            # reshape back to 3D
            batch_size = x.shape[0]
            x_reshaped = x.reshape(batch_size, background_data.shape[1], background_data.shape[2])
            return model_predict_wrapper(x_reshaped)
        
        # 예측 결과 먼저 계산
        print("Computing predictions...")
        predictions = predict_flat(X_test_flat)
        pred_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(pred_classes == y_test)
        print(f"Model Accuracy: {accuracy:.4f}")
        
        # SHAP KernelExplainer 생성 (DeepExplainer 대신 사용)
        print("Creating SHAP KernelExplainer...")
        try:
            # 배경 데이터 서브샘플 (계산량 감소)
            background_subsample = shap.sample(background_data_flat, min(50, background_data_flat.shape[0]))
            
            explainer = shap.KernelExplainer(
                predict_flat,
                background_subsample,
                link="logit"
            )
            
            # SHAP values 계산
            print("Computing SHAP values (this may take a while)...")
            # 테스트 샘플 서브샘플링 (계산량 감소)
            test_subsample = X_test_flat[:min(20, X_test_flat.shape[0])]
            test_labels_subsample = y_test[:min(20, len(y_test))]
            
            shap_values = explainer.shap_values(test_subsample)
            
            # shap_values 처리
            if isinstance(shap_values, list):
                shap_values_list = [np.array(sv) for sv in shap_values]
            else:
                shap_values_list = [np.array(shap_values)]
            
            print("✓ SHAP values computed successfully")
            
        except Exception as e:
            print(f"Warning: SHAP computation failed: {e}")
            print("Using alternative approach: Computing feature importance from gradients...")
            
            # Fallback: 그래디언트 기반 특성 중요도
            shap_values_list = []
            num_classes = 4  # from args.num_class
            
            for class_idx in range(num_classes):
                importance = np.zeros_like(X_test_flat)
                for sample_idx in range(min(10, X_test_flat.shape[0])):  # 10개 샘플만 사용
                    x_sample = X_test_flat[sample_idx:sample_idx+1].copy()
                    x_tensor = torch.from_numpy(x_sample.reshape(1, background_data.shape[1], background_data.shape[2])).float().to(self.device)
                    x_tensor.requires_grad_(True)
                    
                    output = self.model(x_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    
                    prob[0, class_idx].backward(retain_graph=True)
                    
                    if x_tensor.grad is not None:
                        importance[sample_idx] = np.abs(x_tensor.grad.cpu().numpy().reshape(-1))
                
                shap_values_list.append(importance)
        
        # SHAP values 저장
        print("Saving SHAP analysis results...")
        
        # 1. 원본 데이터 저장
        np.save(os.path.join(xai_folder, 'X_test.npy'), X_test)
        np.save(os.path.join(xai_folder, 'y_test.npy'), y_test)
        
        # 2. SHAP values 저장
        for i, sv in enumerate(shap_values_list):
            np.save(os.path.join(xai_folder, f'shap_values_class_{i}.npy'), sv)
        
        # 3. 특성 중요도 분석
        feature_importance = {}
        for class_idx, sv in enumerate(shap_values_list):
            # 평탄화된 특성 중요도
            mean_abs_shap_flat = np.mean(np.abs(sv), axis=0)
            
            # 다시 reshape해서 time-feature별 중요도 계산
            if len(background_data.shape) == 3:
                mean_abs_shap = mean_abs_shap_flat.reshape(background_data.shape[1:])
                mean_abs_shap_per_feature = np.mean(mean_abs_shap, axis=0)  # (num_features,)
            else:
                mean_abs_shap_per_feature = mean_abs_shap_flat
            
            feature_importance[f'class_{class_idx}'] = mean_abs_shap_per_feature.tolist()
        
        # 4. 전체 예측 결과
        all_predictions = predict_flat(X_test_flat)
        all_pred_classes = np.argmax(all_predictions, axis=1)
        
        # 5. 샘플별 설명 생성 (Instance-level explanation)
        print("Generating instance-level explanations...")
        sample_explanations = []
        
        for sample_idx in range(len(X_test)):
            pred_class = all_pred_classes[sample_idx]
            pred_prob = all_predictions[sample_idx]
            true_class = y_test[sample_idx]
            
            # 예측된 클래스의 SHAP values 추출
            pred_class_shap = shap_values_list[pred_class][sample_idx]  # (num_features,)
            
            # 특성별 중요도 계산 (시간차원 평균)
            if len(background_data.shape) == 3:
                pred_class_shap_2d = pred_class_shap.reshape(background_data.shape[1:])
                feature_contributions = np.mean(np.abs(pred_class_shap_2d), axis=0)  # (num_features,)
            else:
                feature_contributions = np.abs(pred_class_shap)
            
            # 상위 5개 중요 특성 찾기
            top_k = 5
            top_feature_indices = np.argsort(feature_contributions)[-top_k:][::-1]
            top_contributions = feature_contributions[top_feature_indices]
            
            # 대조 클래스 (두 번째로 높은 확률의 클래스) 찾기
            contrast_class = np.argsort(pred_prob)[-2]
            contrast_prob = pred_prob[contrast_class]
            
            sample_explanation = {
                'sample_idx': int(sample_idx),
                'predicted_class': int(pred_class),
                'predicted_probability': float(pred_prob[pred_class]),
                'true_class': int(true_class),
                'is_correct': int(pred_class) == int(true_class),
                'all_class_probabilities': {
                    f'class_{i}': float(pred_prob[i]) for i in range(len(pred_prob))
                },
                'contrast_class': int(contrast_class),
                'contrast_probability': float(contrast_prob),
                'top_contributing_features': [
                    {
                        'feature_idx': int(feat_idx),
                        'contribution_magnitude': float(top_contributions[rank])
                    }
                    for rank, feat_idx in enumerate(top_feature_indices)
                ],
                'feature_contribution_scores': feature_contributions.tolist()
            }
            sample_explanations.append(sample_explanation)
        
        # 6. 상세 분석 결과 저장
        analysis_results = {
            'total_samples': len(X_test),
            'num_classes': len(shap_values_list),
            'feature_importance': feature_importance,
            'model_accuracy': float(accuracy),
            'predictions': all_pred_classes.tolist(),
            'true_labels': y_test.tolist(),
            'prediction_probabilities': all_predictions.tolist(),
            'sample_explanations': sample_explanations
        }
        
        with open(os.path.join(xai_folder, 'xai_analysis.json'), 'w') as f:
            json.dump(analysis_results, f, indent=4)
        
        # 샘플 설명을 별도 파일로 저장 (더 쉬운 접근)
        sample_explanations_file = []
        for exp in sample_explanations:
            sample_explanations_file.append(exp)
        
        with open(os.path.join(xai_folder, 'sample_explanations.json'), 'w') as f:
            json.dump(sample_explanations_file, f, indent=4)
        
        # 6. 시각화 - Feature Importance
        print("Creating visualizations...")
        try:
            fig, axes = plt.subplots(1, len(shap_values_list), figsize=(15, 4))
            if len(shap_values_list) == 1:
                axes = [axes]
            
            for class_idx, ax in enumerate(axes):
                importance = feature_importance[f'class_{class_idx}']
                ax.bar(range(len(importance)), importance, color='steelblue', alpha=0.7)
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Mean |SHAP value|')
                ax.set_title(f'Feature Importance - Class {class_idx}')
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(xai_folder, 'feature_importance.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print("✓ Feature importance plot saved")
        except Exception as e:
            print(f"Warning: Could not create feature importance plot: {e}")
        
        # 7. Summary statistics
        summary_text = f"""
SHAP XAI Analysis Summary
========================
Model: {setting}
Total Samples Analyzed: {len(X_test)}
Number of Classes: {len(shap_values_list)}
Model Accuracy on Test Set: {accuracy:.4f}

Input Shape: {X_test.shape}
Flattened Feature Dimension: {X_test_flat.shape[1]}

Feature Importance (Mean |SHAP value|):
"""
        for class_idx in range(len(shap_values_list)):
            importance = feature_importance[f'class_{class_idx}']
            top_features = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)[:5]
            summary_text += f"\n  Class {class_idx}:\n"
            for feat_idx, importance_val in top_features:
                summary_text += f"    Feature {feat_idx}: {importance_val:.6f}\n"
        
        summary_text += f"""
Class Distribution in Predictions:
"""
        for class_idx in range(len(shap_values_list)):
            count = np.sum(all_pred_classes == class_idx)
            summary_text += f"  Class {class_idx}: {count} samples ({100*count/len(all_pred_classes):.1f}%)\n"
        
        with open(os.path.join(xai_folder, 'xai_summary.txt'), 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\n✓ XAI analysis completed. Results saved to: {xai_folder}")
        print("="*50 + "\n")
        
        return analysis_results
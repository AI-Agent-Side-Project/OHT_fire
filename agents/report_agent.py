"""
ReportAgent: LLM 기반 보고서 생성 + RAG (VectorDB)

기능:
- Classification 결과와 XAI SHAP 값을 기반으로 LLM 분석
- 자동 해석 및 권장사항 생성
- RAG를 통한 과거 보고서 참고
- 사용자 조치 입력 기반 최종 보고서 생성
- VectorDB 저장 및 파일 백업
- 구조화된 마크다운 보고서
- 타임스탬프 자동 기록
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import os
from .base_agent import BaseAgent

class ReportAgent(BaseAgent):
    """LLM 기반 보고서 생성 + RAG 에이전트"""

    def __init__(self, 
                 ollama_model: str = "mistral", 
                 ollama_base_url: str = "http://localhost:11434",
                 use_vectordb: bool = True,
                 vectordb_persist_dir: str = "report_db"):
        """초기화"""
        super().__init__(
            agent_id="Report",
            agent_type="Report"
        )
        
        self.report_history = []
        self.llm = None
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.use_vectordb = use_vectordb
        self.vectordb_persist_dir = vectordb_persist_dir
        self.vectorstore = None
        self.last_inference_result = None  # 이전 inference 결과 추적
        
        # Feature 정의 (Feature_Explanation.txt)
        self.feature_definitions = {
            'NTC': '온도 측정값',
            'PM10': '지름이 10μm 이하의 부유먼지(미세먼지) 측정값',
            'PM2.5': '지름이 2.5μm 이하의 먼지(초미세먼지) 측정값',
            'PM1.0': '지름이 1.0μm 이하의 먼지(초미세먼지) 측정값',
            'CT1': '1채널 전류 측정값',
            'CT2': '2채널 전류 측정값',
            'CT3': '3채널 전류 측정값',
            'CT4': '4채널 전류 측정값',
            'ex_temperature': '외부 온도',
            'ex_humidity': '외부 습도',
            'ex_illuminance': '외부 조도'
        }
        
        # Ollama 임포트 (선택적 의존성)
        try:
            from langchain_community.llms import Ollama
            self.Ollama = Ollama
        except ImportError:
            self.logger.warning("LangChain Community not installed. Report generation will be simplified.")
            self.Ollama = None
        
        # VectorDB 임포트 (선택적)
        self.has_vectordb = False
        if self.use_vectordb:
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings import OllamaEmbeddings
                self.Chroma = Chroma
                self.OllamaEmbeddings = OllamaEmbeddings
                self.has_vectordb = True
            except ImportError:
                self.logger.warning("VectorDB dependencies not installed. RAG will be disabled.")
        
        # 히스토리 디렉토리 생성
        self.history_dir = Path("report_history")
        self.history_dir.mkdir(exist_ok=True)

    async def _initialize(self):
        """초기화"""
        if self.Ollama:
            try:
                self.llm = self.Ollama(
                    model=self.ollama_model,
                    base_url=self.ollama_base_url,
                    temperature=0.5
                )
                self.logger.info(f"LLM (Ollama - {self.ollama_model}) 연결 완료")
            except Exception as e:
                self.logger.warning(f"Ollama 연결 실패: {e}. 기본 보고서 생성 모드로 진행합니다.")
                self.logger.info("💡 Ollama를 실행하려면 터미널에서: ollama serve")
                self.llm = None
        else:
            self.logger.info("LLM 미사용 - 기본 보고서 생성 모드")
        
        # VectorDB 초기화
        if self.has_vectordb and self.use_vectordb:
            try:
                Path(self.vectordb_persist_dir).mkdir(exist_ok=True)
                embeddings = self.OllamaEmbeddings(
                    model=self.ollama_model,
                    base_url=self.ollama_base_url
                )
                self.vectorstore = self.Chroma(
                    persist_directory=self.vectordb_persist_dir,
                    embedding_function=embeddings,
                    collection_name="oht_reports"
                )
                self.logger.info("VectorDB (Chroma) 연결 완료")
            except Exception as e:
                self.logger.warning(f"VectorDB 연결 실패: {e}. RAG 기능이 비활성화됩니다.")
                self.vectorstore = None

    async def _run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        보고서 생성 (LLM 기반 + RAG)

        Args:
            input_data: {
                'predicted_class': int,           # Classification 결과 (0-3)
                'top_features': list,             # XAI 상위 특성
                'timestamp': str,                 # 타임스탐프
                'window_data': array,             # 입력 시계열 데이터
                'feature_names': list             # 특성명
            }

        Returns:
            {
                'markdown': str,
                'html': str,
                'json': dict,
                'summary': str,
                'report_id': str,
                'should_report': bool             # Report 생성 여부
            }
        """
        
        # 입력 데이터 추출
        timestamp = input_data.get('timestamp', datetime.now().isoformat())
        predicted_class = input_data.get('predicted_class', 0)
        top_features = input_data.get('top_features', [])
        
        # Classification 결과 → Level 매핑
        class_to_level = {0: 'Normal', 1: 'Grey Zone', 2: 'Warning', 3: 'Danger'}
        level_name = class_to_level.get(predicted_class, '알 수 없음')
        level = predicted_class
        
        # 1. Normal 상태면 report 생성하지 않음
        if predicted_class == 0:
            self.last_inference_result = predicted_class
            return {
                'markdown': '',
                'html': '',
                'json': {},
                'summary': '',
                'report_id': '',
                'should_report': False,
                'reason': 'Normal 상태에서는 report 생성 안함'
            }
        
        # 2. 이전 결과와 같으면 report 생성하지 않음
        if self.last_inference_result is not None and self.last_inference_result == predicted_class:
            return {
                'markdown': '',
                'html': '',
                'json': {},
                'summary': '',
                'report_id': '',
                'should_report': False,
                'reason': f'이전 결과({class_to_level[self.last_inference_result]})와 동일하여 report 생성 안함'
            }
        
        # 현재 결과 저장
        self.last_inference_result = predicted_class
        
        # 3. RAG를 통한 유사 보고서 검색
        similar_reports = []
        if self.vectorstore is not None:
            try:
                # 현재 상태 + 주요 특성을 기반으로 검색
                query = self._create_rag_query(predicted_class, top_features)
                similar_reports = self.vectorstore.similarity_search_with_score(query, k=3)
                self.logger.info(f"유사 보고서 {len(similar_reports)}개 검색됨")
            except Exception as e:
                self.logger.warning(f"RAG 검색 실패: {e}")
        
        # 4. LLM 1차 해석 생성 (초안)
        interpretation_text = await self._generate_interpretation(
            predicted_class,
            top_features,
            similar_reports
        )
        
        # 권장사항 생성
        recommendations = self._generate_recommendations(predicted_class, top_features)
        
        # 5. 1차 보고서 객체 생성
        report = {
            'timestamp': timestamp,
            'level': level,
            'level_name': level_name,
            'predicted_class': predicted_class,
            'top_features': top_features,
            'recommendations': recommendations,
            'interpretation_text': interpretation_text,
            'status': 'draft'  # 초안 상태
        }
        
        # 마크다운 생성
        markdown = self._generate_markdown(report)
        
        # HTML 생성
        html = self._generate_html(report)
        
        # 히스토리에 저장
        report_id = f"report_{timestamp.replace(':', '').replace('-', '').replace('.', '').replace('T', '_')[:20]}"
        self.report_history.append({
            'timestamp': timestamp,
            'level': level,
            'data': report
        })
        if len(self.report_history) > 1000:
            self.report_history.pop(0)
        
        # 1차 보고서를 파일로 저장
        self._save_draft_report(report_id, report)
        
        # 요약 생성
        summary = self._generate_summary(report)
        
        return {
            'markdown': markdown,
            'html': html,
            'json': report,
            'summary': summary,
            'interpretation': interpretation_text,
            'report_id': report_id,
            'should_report': True,
            'recommendations': recommendations
        }

    async def _generate_interpretation(self, 
                                       predicted_class: int, 
                                       top_features: list) -> str:
        """
        LLM을 사용한 해석 생성
        
        Classification 결과와 상위 특성을 입력으로 해석 텍스트 생성
        Feature 정의를 프롬프트에 포함
        """
        
        if not self.llm:
            # LLM 없을 때의 기본 해석
            return self._generate_default_interpretation(
                predicted_class,
                top_features
            )
        
        # LLM 프롬프트 구성
        class_names = ['Normal', 'Grey Zone', 'Warning', 'Danger']
        current_class = class_names[predicted_class]
        
        # 상위 특성 설명 (Feature 정의 포함)
        feature_desc = ""
        if top_features:
            features_info = []
            for f in top_features[:3]:
                feat_name = f['name']
                feat_def = self.feature_definitions.get(feat_name, f'센서 특성: {feat_name}')
                contribution = f['contribution'] * 100
                features_info.append(f"{feat_name} ({feat_def}): 기여도 {contribution:.1f}%")
            feature_desc = "\n주요 영향 특성:\n- " + "\n- ".join(features_info)
        
        # Feature 정의 전체 가이드 (참고용)
        feature_guide = """
[센서 특성 정의]
- NTC: 온도 측정값
- PM10: 지름이 10μm 이하의 부유먼지(미세먼지) 측정값
- PM2.5: 지름이 2.5μm 이하의 먼지(초미세먼지) 측정값
- PM1.0: 지름이 1.0μm 이하의 먼지(초미세먼지) 측정값
- CT1, CT2, CT3, CT4: 1-4채널 전류 측정값
- ex_temperature: 외부 온도
- ex_humidity: 외부 습도
- ex_illuminance: 외부 조도"""
        
        prompt = f"""당신은 OHT(Overhead Hoist Transfer) 시스템의 센서 모니터링 전문가입니다.

{feature_guide}

다음 센서 모니터링 데이터를 분석하고 간단하고 명확한 해석을 제공하세요:

현재 상태: {current_class}
{feature_desc}

다음 형식으로 한국어로 2-3문장의 간결한 해석을 작성하세요:
1. 현재 상태 요약 (센서 정의를 고려하여)
2. 주요 원인 (어떤 센서가 문제인지)
3. 권장 조치 (구체적인 검사/조정 항목)

마크다운 포맷이나 ** ** 없이 순수 텍스트만 작성하세요."""
        
        try:
            # Ollama는 invoke 사용 (ainvoke 지원 제한)
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            self.logger.warning(f"LLM 호출 실패: {e}. 기본 해석으로 대체합니다.")
            return self._generate_default_interpretation(
                predicted_class,
                top_features
            )

    def _generate_default_interpretation(self, 
                                        predicted_class: int,
                                        top_features: list) -> str:
        """Rule 기반 기본 해석 (Feature 정의 포함)"""
        
        class_desc = {
            0: "Normal 상태로 감지되었습니다.",
            1: "Grey Zone 상태입니다. 모니터링을 강화해주세요.",
            2: "Warning 상태입니다. 조사와 대응이 필요합니다.",
            3: "Danger 상태입니다. 즉시 대응이 필요합니다."
        }
        
        description = class_desc.get(predicted_class, "상태 불명확")
        
        if top_features and len(top_features) > 0:
            main_feature = top_features[0]['name']
            feat_def = self.feature_definitions.get(main_feature, main_feature)
            contrib = top_features[0]['contribution'] * 100
            cause = f"주요 원인은 {main_feature}({feat_def}, 기여도 {contrib:.1f}%)으로 분석됩니다."
        else:
            cause = ""
        
        return f"{description} {cause}"

    def _generate_recommendations(self, predicted_class: int, top_features: list) -> List[str]:
        """권장사항 생성"""
        
        base_recommendations = {
            0: [
                "정상 상태를 유지하기 위해 정기적인 모니터링 지속",
                "현재 설정값 기준으로 정상 범위 유지 중"
            ],
            1: [
                "센서 데이터 이상 여부 확인 필요",
                "설정 파라미터 검토 및 조정 고려",
                "추가 진단 정보 수집"
            ],
            2: [
                "전문가 상담 및 검증 필요",
                "해당 센서에 대한 교정 점검",
                "시스템 파라미터 재조정 검토"
            ],
            3: [
                "즉시 현장 확인 및 대응 필요",
                "응급 매뉴얼 참조",
                "관련 부서 긴급 연락"
            ]
        }
        
        recs = base_recommendations.get(predicted_class, [])
        
        # 상위 특성 기반 추가 권장사항
        if top_features and len(top_features) > 0:
            main_feature = top_features[0]['name']
            recs.append(f"{main_feature} 센서 상태 우선 점검")
        
        return recs[:3]  # 최대 3개만 반환

    def _generate_markdown(self, report: Dict) -> str:
        """마크다운 형식 보고서 (불필요한 정보 제거)"""
        
        md = []
        
        # 헤더
        level_emoji = {0: '✅', 1: '⚠️', 2: '🔔', 3: '🚨'}.get(report['level'], '❓')
        md.append(f"# {level_emoji} {report['level_name']} - 실시간 모니터링 보고서")
        md.append("")
        
        # 기본 정보
        md.append("## 📊 기본 정보")
        md.append(f"- **시간**: {report['timestamp']}")
        md.append(f"- **상태**: {report['level_name']}")
        md.append("")
        
        # LLM 해석
        md.append("## 🔍 분석 해석")
        md.append(f"> {report['interpretation_text']}")
        md.append("")
        
        # 상위 특성
        if report.get('top_features'):
            md.append("## ⭐ 상위 영향 특성")
            md.append("| 순위 | 특성명 | 기여도 |")
            md.append("|------|--------|--------|")
            for i, feat in enumerate(report['top_features'][:3], 1):
                md.append(
                    f"| {i} | {feat['name']} | "
                    f"{feat['contribution']*100:.1f}% |"
                )
            md.append("")
        
        # 권장사항
        if report.get('recommendations'):
            md.append("## 💡 권장사항")
            for i, rec in enumerate(report['recommendations'], 1):
                md.append(f"{i}. {rec}")
            md.append("")
        
        return "\n".join(md)

    def _generate_html(self, report: Dict) -> str:
        """HTML 형식 보고서 (불필요한 정보 제거)"""
        
        level_color = {0: 'green', 1: 'orange', 2: 'orange', 3: 'red'}.get(report['level'], 'gray')
        level_emoji = {0: '✅', 1: '⚠️', 2: '🔔', 3: '🚨'}.get(report['level'], '❓')
        
        features_html = ""
        if report.get('top_features'):
            features_html = """
            <h2>⭐ 상위 영향 특성</h2>
            <table style="border-collapse: collapse; width: 100%;">
                <tr style="border-bottom: 1px solid #ddd;">
                    <th style="padding: 8px; text-align: left;">특성</th>
                    <th style="padding: 8px; text-align: left;">기여도</th>
                </tr>
            """
            for feat in report['top_features'][:3]:
                features_html += f"""
                <tr style="border-bottom: 1px solid #ddd;">
                    <td style="padding: 8px;">{feat['name']}</td>
                    <td style="padding: 8px;">{feat['contribution']*100:.1f}%</td>
                </tr>
                """
            features_html += "</table>"
        
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 20px;">
            <h1 style="color: {level_color};">
                {level_emoji} {report['level_name']}
            </h1>
            
            <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px;">
                <p><strong>시간:</strong> {report['timestamp']}</p>
                <p><strong>상태:</strong> {report['level_name']}</p>
            </div>
            
            <h2>🔍 분석 해석</h2>
            <p>{report['interpretation_text']}</p>
            
            {features_html}
            
            <h2>💡 권장사항</h2>
            <ul>
                {''.join([f'<li>{rec}</li>' for rec in report.get('recommendations', [])])}
            </ul>
        </div>
        """
        
        return html

    def _generate_summary(self, report: Dict) -> str:
        """한 줄 요약"""
        
        timestamp = report['timestamp'].split('T')[1].split('+')[0] if 'T' in report['timestamp'] else report['timestamp']
        level_name = report['level_name']
        
        if report.get('top_features'):
            feature_names = ", ".join([f['name'] for f in report['top_features'][:2]])
            return f"[{timestamp}] {level_name}: {feature_names} 영향"
        else:
            return f"[{timestamp}] {level_name}"

    # ==================== 새로운 RAG 관련 메서드 ====================
    
    def _create_rag_query(self, predicted_class: int, top_features: list) -> str:
        """RAG 검색용 쿼리 생성"""
        class_names = ['Normal', 'Grey Zone', 'Warning', 'Danger']
        class_name = class_names[predicted_class]
        
        features_str = ""
        if top_features:
            features_str = ", ".join([f['name'] for f in top_features[:3]])
        
        return f"상태: {class_name}, 영향 특성: {features_str}"
    
    async def _generate_interpretation(self,
                                       predicted_class: int,
                                       top_features: list,
                                       similar_reports: list = None) -> str:
        """
        LLM을 사용한 해석 생성 (RAG 정보 포함)
        
        Classification 결과와 상위 특성, 유사 보고서를 입력으로 해석 텍스트 생성
        """
        
        if not self.llm:
            return self._generate_default_interpretation(predicted_class, top_features)
        
        # LLM 프롬프트 구성
        class_names = ['Normal', 'Grey Zone', 'Warning', 'Danger']
        current_class = class_names[predicted_class]
        
        # 상위 특성 설명
        feature_desc = ""
        if top_features:
            features_info = []
            for f in top_features[:3]:
                feat_name = f['name']
                feat_def = self.feature_definitions.get(feat_name, f'센서 특성: {feat_name}')
                contribution = f['contribution'] * 100
                features_info.append(f"{feat_name} ({feat_def}): 기여도 {contribution:.1f}%")
            feature_desc = "\n주요 영향 특성:\n- " + "\n- ".join(features_info)
        
        # 유사 보고서 정보
        similar_context = ""
        if similar_reports:
            similar_context = "\n\n과거 유사 사례:\n"
            for i, (doc, score) in enumerate(similar_reports[:2], 1):
                similar_context += f"- 케이스 {i}: {doc.page_content[:100]}...\n"
        
        feature_guide = """
[센서 특성 정의]
- NTC: 온도 측정값
- PM10: 지름이 10μm 이하의 부유먼지(미세먼지) 측정값
- PM2.5: 지름이 2.5μm 이하의 먼지(초미세먼지) 측정값
- PM1.0: 지름이 1.0μm 이하의 먼지(초미세먼지) 측정값
- CT1, CT2, CT3, CT4: 1-4채널 전류 측정값
- ex_temperature: 외부 온도
- ex_humidity: 외부 습도
- ex_illuminance: 외부 조도"""
        
        prompt = f"""당신은 OHT(Overhead Hoist Transfer) 시스템의 센서 모니터링 전문가입니다.

{feature_guide}

현재 센서 모니터링 데이터를 분석하고 간단하고 명확한 해석을 제공하세요:

현재 상태: {current_class}
{feature_desc}
{similar_context}

다음 형식으로 한국어로 2-3문장의 간결한 초안 해석을 작성하세요:
1. 현재 상태 요약 (센서 정의를 고려하여)
2. 주요 원인 (어떤 센서가 문제인지)
3. 권장 조치 (구체적인 검사/조정 항목)

마크다운 포맷이나 ** ** 없이 순수 텍스트만 작성하세요.
사용자의 실제 조치 입력을 기다리고 있으니, 조치는 권장사항으로 제시하세요."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            self.logger.warning(f"LLM 호출 실패: {e}. 기본 해석으로 대체합니다.")
            return self._generate_default_interpretation(predicted_class, top_features)
    
    def _save_draft_report(self, report_id: str, report: Dict) -> str:
        """1차 보고서를 파일로 저장"""
        
        draft_dir = self.history_dir / "draft"
        draft_dir.mkdir(exist_ok=True)
        
        # JSON 저장
        draft_file = draft_dir / f"{report_id}.json"
        with open(draft_file, 'w', encoding='utf-8') as f:
            # Report JSON 직렬화 (datetime 객체 제거)
            report_to_save = {
                'timestamp': report['timestamp'],
                'level': report['level'],
                'level_name': report['level_name'],
                'predicted_class': report['predicted_class'],
                'interpretation_text': report['interpretation_text'],
                'recommendations': report['recommendations'],
                'top_features': report.get('top_features', []),
                'status': 'draft'
            }
            json.dump(report_to_save, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"1차 보고서 저장: {draft_file}")
        return str(draft_file)
    
    async def finalize_report(self, report_id: str, user_action: str) -> Dict[str, Any]:
        """
        사용자 조치 입력 후 최종 보고서 생성
        
        Args:
            report_id: 보고서 ID
            user_action: 사용자가 입력한 실제 조치
        
        Returns:
            최종 보고서 정보
        """
        
        # 1차 보고서 로드
        draft_file = self.history_dir / "draft" / f"{report_id}.json"
        if not draft_file.exists():
            self.logger.error(f"보고서를 찾을 수 없음: {draft_file}")
            return {'success': False, 'error': '보고서를 찾을 수 없음'}
        
        with open(draft_file, 'r', encoding='utf-8') as f:
            draft_report = json.load(f)
        
        # LLM을 통해 최종 보고서 생성
        final_interpretation = ""
        if self.llm:
            class_names = ['Normal', 'Grey Zone', 'Warning', 'Danger']
            current_class = class_names[draft_report['predicted_class']]
            
            prompt = f"""당신은 OHT(Overhead Hoist Transfer) 시스템의 센서 모니터링 전문가입니다.

다음의 1차 분석과 실제 조치 결과를 바탕으로 최종 보고서를 작성하세요:

상태: {current_class}
1차 분석: {draft_report['interpretation_text']}
실제 조치: {user_action}

한국어로 2-3문장의 최종 보고서를 작성하세요:
1. 상태 요약
2. 원인 분석
3. 실시한 조치와 결과

마크다운 포맷 없이 순수 텍스트만 작성하세요."""
            
            try:
                final_interpretation = self.llm.invoke(prompt).strip()
            except Exception as e:
                self.logger.warning(f"최종 LLM 호출 실패: {e}")
                final_interpretation = f"상태: {current_class}\n조치: {user_action}"
        else:
            final_interpretation = f"실제 조치: {user_action}"
        
        # 최종 보고서 생성
        final_report = {
            **draft_report,
            'user_action': user_action,
            'final_interpretation': final_interpretation,
            'status': 'final',
            'finalized_at': datetime.now().isoformat()
        }
        
        # 최종 보고서 파일 저장
        final_dir = self.history_dir / "final"
        final_dir.mkdir(exist_ok=True)
        final_file = final_dir / f"{report_id}_final.json"
        
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"최종 보고서 저장: {final_file}")
        
        # VectorDB에 저장
        if self.vectorstore is not None:
            try:
                # 검색용 텍스트 생성
                query_text = f"""
                상태: {final_report['level_name']}
                시간: {final_report['timestamp']}
                원인: {final_report['final_interpretation']}
                조치: {user_action}
                특성: {', '.join([f['name'] for f in final_report.get('top_features', [])])}
                """
                
                self.vectorstore.add_texts(
                    texts=[query_text],
                    metadatas=[{
                        'report_id': report_id,
                        'level': final_report['level_name'],
                        'timestamp': final_report['timestamp'],
                        'action': user_action,
                        'source': 'final_report'
                    }]
                )
                self.logger.info(f"최종 보고서를 VectorDB에 저장")
            except Exception as e:
                self.logger.warning(f"VectorDB 저장 실패: {e}")
        
        return {
            'success': True,
            'report_id': report_id,
            'final_report': final_report,
            'file_path': str(final_file)
        }

    async def _cleanup(self):
        """리소스 정리"""
        if self.vectorstore is not None:
            try:
                self.vectorstore.persist()
                self.logger.info("VectorDB 저장 완료")
            except Exception as e:
                self.logger.warning(f"VectorDB 저장 실패: {e}")


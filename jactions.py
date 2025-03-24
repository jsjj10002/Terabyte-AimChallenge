from spacy import load
from transformers import pipeline
import sqlite3
import json
from typing import Any, Text, Dict, List, Optional, Tuple, Union
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import logging
from datetime import datetime
import re
import stanza
stanza.download('ko')
nlp = stanza.Pipeline(lang='ko', processors='tokenize,pos,lemma,depparse', tokenize_pretokenized=False, verbose=False, tokenize_no_ssplit=True)
# 로거 설정
import logging
import threading

class LoggerSetup:
    _logger = None
    _lock = threading.Lock()

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            with cls._lock:  # 다중 스레드 환경에서 동시성 문제 방지
                if cls._logger is None:
                    cls._logger = cls._setup_logger()
        return cls._logger

    @staticmethod
    def _setup_logger():
        """로깅 설정을 초기화하고 logger 인스턴스를 반환"""
        logger = logging.getLogger("CoffeeBot")
        logger.setLevel(logging.DEBUG)

        # 기존 핸들러가 없다면 추가
        if not logger.handlers:
            # 콘솔 출력 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)

            # 로그 포맷 설정
            formatter = logging.Formatter(
                "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)

            # 핸들러 추가
            logger.addHandler(console_handler)

            # (선택 사항) 파일 핸들러
            file_handler = logging.FileHandler("coffeebot.log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    
# 매세지 생성 매서드
class MessageManager:
    """메시지 생성과 관리를 담당하는 클래스"""
    
    _message_templates = {
        # 주문 관련 메시지
        "order_confirmation": "주문이 확인되었습니다: {items}",
        "order_modified": "{item}이(가) {modification_type}되었습니다",
        "order_cancelled": "주문이 취소되었습니다",
        "order_completed": "주문이 완료되었습니다. 총 주문 내역: {summary}",
        
        # 옵션 변경 관련 메시지
        "size_changed": "{drink_type}의 사이즈가 {size}로 변경되었습니다",
        "temperature_changed": "{drink_type}의 온도가 {temperature}로 변경되었습니다",
        "option_added": "{drink_type}에 {additional_options} 옵션이 추가되었습니다",
        "option_removed": "{drink_type}에서 {additional_options} 옵션이 제거되었습니다",
        
        # 에러 메시지
        "error_not_found": "{item}을(를) 찾을 수 없습니다",
        "error_invalid_temperature": "{drink_type}는(은) {temperature}로 주문할 수 없습니다",
        "error_general": "처리 중 오류가 발생했습니다: {error_message}",
        
        # 현재 상태 메시지
        "current_order": "현재 주문 내역: {order_summary}",
        "additional_option_query": "다른 추가 옵션이 필요하신가요?"
    }

    @classmethod
    def create_message(cls, message_type: str, **kwargs) -> str:
        """
        지정된 메시지 타입과 매개변수를 사용하여 메시지를 생성함
        
        Args:
            message_type (str): 메시지 템플릿의 키값
            **kwargs: 메시지 템플릿에 들어갈 변수들
            
        Returns:
            str: 생성된 메시지
            
        Raises:
            KeyError: 존재하지 않는 메시지 타입일 경우
        """
        logger = LoggerSetup.get_logger()
        
        try:
            template = cls._message_templates[message_type]
            message = template.format(**kwargs)
            logger.debug(f"메시지 생성 - 타입: {message_type}, 내용: {message}")
            return message
        except KeyError:
            logger.error(f"존재하지 않는 메시지 타입: {message_type}")
            return cls._message_templates["error_general"].format(
                error_message=f"알 수 없는 메시지 타입: {message_type}"
            )
        except Exception as e:
            logger.error(f"메시지 생성 중 오류 발생: {str(e)}")
            return cls._message_templates["error_general"].format(
                error_message="메시지 생성 실패"
            ) 

# 표준화 매서드
class DrinkStandardizer:
    """음료 관련 데이터 표준화를 담당하는 클래스"""
    
    _logger = LoggerSetup.get_logger()
    
    # 표준화 매핑 사전
    _mapping = {
        "drink_type": {
            # 음료 종류 매핑
            "카페라테": "카페라떼",
            "카페라뗴": "카페라떼",
            "레모네이드" : "레몬에이드",
            "카라멜마키아또" : "카라멜마끼아또",
            "아보카도" : "아포카토",
            "키즈스" : "키위주스",
            "초콜릿" : "초콜릿라떼",
            "초콜릿대" : "초콜릿라떼",
            "바닐라떼" : "바닐라라떼",
            "카라멜막혔더" : "카라멜마끼아또",
            "복숭아ost" : "복숭아아이스티",
            "말자라때" : "말차라떼",
            "바닐라레떼" : "바닐라라떼",
            "아포가토" : "아포카토",
            "복숭아아이스크림" : "복숭아아이스티",
            "허벅지" : "허브티",
            "에스페로" : "에스프레소",
            "다기스무디" : "딸기스무디",
            "망고스머리" : "망고스무디",
            "토마토소스" : "토마토주스",
            "망고스뮤비" : "망고스무디",
            "쿠킹크림" : "쿠키앤크림",
            "쿠킹그림" : "쿠키앤크림",
            "쿠앤크" : "쿠키앤크림",
            "카페북한" : "카페모카",
            "tv스투스" : "키위주스",
            "Tv스투스" : "키위주스",
            "TV스투스" : "키위주스",
            "말잘할때" : "말차라떼",
            "허버트" : "허브티",
            "tv쥬스" : "키위주스",
            "Tv쥬스" : "키위주스",
            "TV쥬스" : "키위주스",
            "아프리카" : "아포카토",
            "마찰할때" : "말차라떼",
            "말찾았대" : "말차라떼",
            "라벨마끼아또" : "카라멜마끼아또",
            "카메라맡기어도" : "카라멜마끼아또",
            "복숭아st" : "복숭아아이스티",
            "복숭아St" : "복숭아아이스티",
            "복숭아ST" : "복숭아아이스티",
            "복숭아에스티" : "복숭아아이스티",
            "복숭아하이스틸" : "복숭아아이스티",
            "호텔" : "허브티",
            "말잘했다" : "말차라떼",
            "카프치노" : "카푸치노",
            "카라멜마끼야또" : "카라멜마끼아또",
            "라떼" : "카페라떼",
            "라뗴" : "카페라떼",
            "라때" : "카페라떼",
            "모카" : "카페모카",
            "카라멜" : "카라멜마끼아또",
            "마기야도" : "카라멜마끼아또",
            "카라멜마기야도" : "카라멜마끼아또"
        },
        "temperature": {
            # 온도 매핑
            "차가운": "아이스",
            "시원한": "아이스",
            "차갑게": "아이스", 
            "시원하게": "아이스",
            "뜨겁게": "핫", 
            "따뜻하게": "핫", 
            "핫": "핫", 
            "뜨거운": "핫", 
            "따뜻한": "핫", 
            "뜨뜻한": "아이스", 
            "하수": "핫", 
            "hot": "핫"
        },
        "size": {
            # 사이즈 매핑
            "큰": "라지",
            "크게": "라지", 
            "라의": "라지", 
            "라디오": "라지", 
            "라디": "라지",
            "엑스라지": "엑스라지",
            "엑스라이즈": "엑스라지", 
            "제1 큰": "엑스라지", 
            "가장 큰": "엑스라지", 
            "제1 크게": "엑스라지", 
            "맥시멈": "엑스라지",
            "미디움": "미디움",
            "중간": "미디움", 
            "기본": "미디움", 
            "톨": "미디움", 
            "비디오": "미디움", 
            "토": "미디움",
            "보통": "미디움"
        },
        "additional_options": {
            # 옵션 매핑
            "샤타나": "샷",
            "4추가": "샷",
            "샤츠": "샷",
            "셔츠": "샷",
            "사추": "샷",
            "샷추가": "샷",
            "카라멜실업": "카라멜시럽",
            "실룩실룩": "카라멜시럽",
            "가라멜시럽": "카라멜시럽",
            "카라멜시로": "카라멜시럽",
            "바닐라실업": "바닐라시럽",
            "비비크림": "휘핑크림"
        },
        "take": {
            # 테이크 매핑
            "테이크아웃": "포장",
            "들고": "포장",
            "가져": "포장",
            "먹을": "포장",
            "마실": "포장",
            "아니요": "포장",
            "먹고": "매장",
            "여기": "매장",
            "이곳": "매장",
            "네": "매장"
        },
        "quantity": {
            # 잔 수 매핑
            "한": 1, "하나": 1, "1": 1,"원": 1,
            "두": 2, "둘": 2, "2": 2, "투": 2, "더블": 2,
            "세": 3, "셋": 3, "3": 3, "쓰리": 3, "트리플": 3,
            "네": 4, "넷": 4, "4": 4, "포": 4, 
            "다섯": 5, "5": 5, "파이브": 5, 
            "여섯": 6, "6": 6, "식스": 6, 
            "일곱": 7, "7": 7, "세븐": 7, 
            "여덟": 8, "8": 8, "에잇": 8, 
            "아홉": 9, "9": 9, "나인": 9,
            "열": 10, "10": 10
        }
    }
    
    # 음료별 제약조건
    _drink_constraints = {
        "ice_only": ["토마토주스", "키위주스", "망고스무디", "딸기스무디", "레몬에이드", "복숭아아이스티"] ,
        "hot_only": ["허브티"]
    }
    
    @classmethod
    def standardize(cls, entity_type: str, value: str) -> str:
        """엔티티 값을 표준화된 형식으로 변환"""
        try:
            standardized = cls._mapping.get(entity_type, {}).get(value, value)
            cls._logger.debug(f"표준화: {entity_type} - {value} -> {standardized}")
            return standardized
        except Exception as e:
            cls._logger.error(f"표준화 중 오류 발생: {str(e)}")
            return value

class IntentSplitter:
    """문장을 개별 음료 주문으로 분리하는 클래스"""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger()
        self.standardizer = DrinkStandardizer()
        
    def _find_drink_groups(self, text: str, entities: List[Dict]) -> List[Dict]:
        """음료 그룹을 찾아서 반환"""
        drink_groups = []
        
        # 음료 타입 엔티티들을 찾음
        drink_entities = [e for e in entities if e['entity'] == 'drink_type']
        
        # 각 음료별로 그룹 생성
        for drink_entity in drink_entities:
            group = {
                'drink_type': drink_entity['value'],
                'start': drink_entity['start'],
                'end': drink_entity['end'],
                'modifiers': []  # 이 음료와 연관된 수식어들
            }
            drink_groups.append(group)
            
        return drink_groups

    def _assign_modifiers(self, text: str, entities: List[Dict], drink_groups: List[Dict]):
        """각 음료 그룹에 수식어 할당"""
        for entity in entities:
            if entity['entity'] in ['temperature', 'size', 'quantity']:
                # 가장 가까운 음료 그룹 찾기
                closest_group = min(
                    drink_groups,
                    key=lambda g: abs(g['start'] - entity['start'])
                )
                closest_group['modifiers'].append(entity)

    def split_intents(self, text: str, entities: List[Dict]) -> Tuple[List[str], List[List[Dict]]]:
        """하나의 복합 주문을 여러 개의 단순 주문으로 분리"""
        try:
            # 1. 음료 그룹 찾기
            drink_groups = self._find_drink_groups(text, entities)

            # 2. 구분자 기반 분리
            split_orders = self._split_by_delimiters(text, drink_groups)
            self.logger.debug(f"분리된 주문 세그먼트: {split_orders}")

            # 3. 각 그룹을 개별 주문 문장으로 변환
            split_texts = []
            split_entities = []
            for order in split_orders:
                segment_text = order['text'].strip()
                segment_entities = self._extract_entities_for_segment(segment_text, entities, text)
                split_texts.append(segment_text)
                split_entities.append(segment_entities)
             
            self.logger.debug(f"분리된 텍스트: {split_texts}")
            self.logger.debug(f"분리된 엔티티: {split_entities}")
            return split_texts, split_entities

        except Exception as e:
            self.logger.error(f"주문 분리 중 오류 발생: {str(e)}")
            raise

    def _split_by_delimiters(self, text: str, drink_groups: List[Dict]) -> List[Dict]:
        """구분자를 기반으로 주문 분리"""
        # 구분자 패턴 정의
        delimiters = {
             'sequence': ['그리고', '또한', '그다음', '그리고나서', '하나는', '또 하나는', '다른 하나는'],
             'contrast': ['하나는', '또 하나는', '다른 하나는', '나머지는'],
             'addition': ['추가로', '더', '또'],
             'separation': ['이랑', '랑', '와', '과', '하고']
        }
        
        # 정규표현식 패턴 생성
        pattern = '|'.join([re.escape(d) for group in delimiters.values() for d in group])
        
        # 초기 분리
        segments = re.split(f'({pattern})', text)
        segments = [s.strip() for s in segments if s.strip()]
        
        # 분리된 세그먼트를 주문 그룹으로 변환
        orders = []
        current_order = {'text': '', 'modifiers': []}

        for segment in segments:
            # 구분자인 경우
            is_delimiter = any(segment in group for group in delimiters.values())
             
            if is_delimiter:
                if current_order['text']:
                    orders.append(current_order)
                current_order = {'text': '', 'modifiers': []}
            else:
                current_order['text'] += ' ' + segment
         
        if current_order['text']:
            orders.append(current_order)
             
        return orders

    def _extract_entities_for_segment(self, segment_text: str, all_entities: List[Dict], original_text: str) -> List[Dict]:
        """주어진 세그먼트 텍스트에 해당하는 엔티티를 추출"""
        segment_offset = original_text.find(segment_text)
        if segment_offset == -1:
            self.logger.warning(f"세그먼트를 원본 텍스트에서 찾을 수 없습니다: {segment_text}")
            segment_offset = 0
         
        # 해당 세그먼트에 속하는 엔티티 추출
        segment_entities = [
            {
                **entity,
                'start': entity['start'] - segment_offset,
                'end': entity['end'] - segment_offset
            }
            for entity in all_entities
            if segment_offset <= entity['start'] < segment_offset + len(segment_text)
        ]
         
        return segment_entities

class Drink:
    """음료 객체를 표현하는 클래스"""
    
    def __init__(
        self,
        drink_type: str = None,
        temperature: str = "핫",
        size: str = "미디움",
        quantity: int = 1,
        additional_options: List[str] = None,
        order_id: str = None
    ):
        self.drink_type = drink_type
        self.temperature = temperature
        self.size = size
        self.quantity = quantity
        self.additional_options = additional_options if additional_options else []
        self.order_id = order_id

    def __str__(self) -> str:
        """객체의 문자열 표현을 반환"""
        options = f", 추가 옵션: {', '.join(self.additional_options)}" if self.additional_options else ""
        return f"{self.temperature} {self.drink_type} {self.size} 사이즈 {self.quantity}잔{options}"

    def add_option(self, option: str):
        """추가 옵션을 객체에 추가"""
        if option not in self.additional_options:
            self.additional_options.append(option)

    def update_quantity(self, quantity: int):
        """음료 수량을 업데이트"""
        if quantity > 0:
            self.quantity = quantity
        else:
            raise ValueError("수량은 0보다 커야 합니다.")

    def validate(self):
        """음료 객체의 유효성을 검사"""
        if not self.drink_type:
            raise ValueError("음료 타입이 설정되지 않았습니다.")
        if self.temperature not in ["핫", "아이스"]:
            raise ValueError(f"잘못된 온도 설정: {self.temperature}")
        if self.size not in ["스몰", "미디움", "라지"]:
            raise ValueError(f"잘못된 사이즈 설정: {self.size}")

    def to_dict(self) -> dict:
        """음료 객체를 딕셔너리로 변환"""
        return {
            "drink_type": self.drink_type,
            "temperature": self.temperature,
            "size": self.size,
            "quantity": self.quantity,
            "additional_options": self.additional_options,
            "order_id": self.order_id
        }

    @classmethod
    def from_dict(cls, data: dict):
        """딕셔너리를 기반으로 음료 객체 생성"""
        return cls(
            drink_type=data.get("drink_type"),
            temperature=data.get("temperature", "핫"),
            size=data.get("size", "미디움"),
            quantity=data.get("quantity", 1),
            additional_options=data.get("additional_options", []),
            order_id=data.get("order_id")
        )


# 음료 객제 매핑
class DrinkMapper:
    """RASA 엔티티를 음료 객체로 매핑하는 클래스"""

    def __init__(self):
        self.logger = LoggerSetup.get_logger()
        self.standardizer = DrinkStandardizer()
        self.intent_splitter = IntentSplitter()

    def analyze_dependency(self, text: str) -> List[Dict]:
        """텍스트의 의존 구문을 분석"""
        doc = nlp(text)
        parsed_results = []
        for sent in doc.sentences:
            for word in sent.words:
                word_info = {
                    'id': word.id,
                    'text': word.text,
                    'lemma': word.lemma,
                    'upos': word.upos,
                    'xpos': word.xpos,
                    'head': word.head,
                    'deprel': word.deprel,
                    'start_char': word.start_char,
                    'end_char': word.end_char,
                }
                parsed_results.append(word_info)
        return parsed_results

    def _is_quantity_related_to_option(self, parsed: List[Dict], quantity_entity: Dict, option_entity: Dict) -> bool:
        """의존구문 분석 결과를 바탕으로 수량이 옵션과 관련 있는지 확인"""
        quantity_word = next((word for word in parsed if word['start_char'] == quantity_entity['start'] and word['end_char'] == quantity_entity['end']), None)
        option_word = next((word for word in parsed if word['start_char'] == option_entity['start'] and word['end_char'] == option_entity['end']), None)

        if not quantity_word or not option_word:
            return False

        # 의존 관계 및 동일한 동사 의존 여부 확인
        return (
            quantity_word['head'] == option_word['id'] or
            option_word['head'] == quantity_word['id'] or
            (quantity_word['head'] == option_word['head'] and quantity_word['deprel'] in ['nummod', 'det'] and option_word['deprel'] in ['obj', 'compound'])
        )

    def _get_option_quantities(self, parsed: List[Dict], entities: List[Dict]) -> Dict[str, int]:
        """추가 옵션별 수량을 분석하여 반환"""
        option_quantities = {}
        option_entities = [e for e in entities if e['entity'] == 'additional_options']

        for option_entity in option_entities:
            count = 1
            for quantity_entity in entities:
                if quantity_entity['entity'] == 'quantity' and self._is_quantity_related_to_option(parsed, quantity_entity, option_entity):
                    count = self.standardizer.standardize('quantity', quantity_entity['value'])
                    break

            standardized_option = self.standardizer.standardize('additional_options', option_entity['value'])
            option_quantities[standardized_option] = option_quantities.get(standardized_option, 0) + count

        return option_quantities

    def extract_entities(self, message_data: Dict) -> List[Dict]:
        """RASA 메시지 데이터에서 유효한 엔티티만 추출"""
        entities = message_data.get("entities", [])
        valid_entities = [entity for entity in entities if entity.get("extractor") != "DIETClassifier"]
        self.logger.debug(f"추출된 엔티티: {valid_entities}")
        return valid_entities

    def create_drink_from_entities(self, entities: List[Dict], user_text: str) -> List[Drink]:
        """엔티티 리스트와 사용자 텍스트로부터 음료 객체 생성"""
        try:
            split_texts, split_entities = self.intent_splitter.split_intents(user_text, entities)
            drinks = []

            for segment_text, order_entities in zip(split_texts, split_entities):
                drink = Drink()
                parsed = self.analyze_dependency(segment_text) if segment_text else []
                option_quantities = self._get_option_quantities(parsed, order_entities)

                for entity in sorted(order_entities, key=lambda x: x.get("start", 0)):
                    entity_type = entity.get("entity")
                    value = entity.get("value")
                    if entity_type == "quantity":
                        drink.quantity = self.standardizer.standardize(entity_type, value)
                    elif entity_type == "drink_type":
                        drink.drink_type = self.standardizer.standardize(entity_type, value)
                    elif entity_type == "temperature":
                        drink.temperature = self.standardizer.standardize(entity_type, value)
                    elif entity_type == "size":
                        drink.size = self.standardizer.standardize(entity_type, value)
                    elif entity_type == "additional_options":
                        drink.additional_options.append(self.standardizer.standardize(entity_type, value))

                # 옵션 수량 반영
                for option, count in option_quantities.items():
                    drink.additional_options.extend([option] * count)

                self.logger.info(f"음료 객체 생성 완료: {str(drink)}")
                drinks.append(drink)

            return drinks
        except Exception as e:
            self.logger.error(f"음료 객체 생성 중 오류 발생: {str(e)}")
            raise ValueError(f"음료 객체 생성 실패: {str(e)}")

    def validate_drink(self, drinks: List[Drink]) -> bool:
        """음료 객체의 유효성 검사"""
        try:
            for drink in drinks:
                if not drink.drink_type:
                    raise ValueError("음료 종류가 지정되지 않았습니다.")
                self.logger.info(f"음료 유효성 검사 통과: {str(drink)}")
            return True
        except Exception as e:
            self.logger.error(f"음료 유효성 검사 실패: {str(e)}")
            raise



class OrderStorage:
    """SQL 기반 주문 저장소를 관리하는 클래스"""
    
    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self._db_path = 'coffee_orders.db'
        self._init_database()

    def _init_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                # 주문 테이블 생성
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id TEXT PRIMARY KEY,
                        drink_type TEXT NOT NULL,
                        temperature TEXT NOT NULL,
                        size TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                # 옵션 테이블 생성
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS order_options (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT NOT NULL,
                        option_name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (order_id) REFERENCES orders(order_id)
                    )
                ''')
                conn.commit()
                self._logger.info("데이터베이스 초기화 완료")
        except Exception as e:
            self._logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
            raise

    def _generate_order_id(self) -> str:
        """주문 ID 생성"""
        return f"order_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def add_order(self, drinks: List[Drink]) -> List[str]:
        """음료 주문을 저장"""
        try:
            order_ids = []
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                for drink in drinks:
                    for _ in range(drink.quantity):
                        order_id = self._generate_order_id()
                        order_ids.append(order_id)
                        cursor.execute('''
                            INSERT INTO orders (order_id, drink_type, temperature, size)
                            VALUES (?, ?, ?, ?)
                        ''', (order_id, drink.drink_type, drink.temperature, drink.size))
                        for option in drink.additional_options:
                            cursor.execute('''
                                INSERT INTO order_options (order_id, option_name)
                                VALUES (?, ?)
                            ''', (order_id, option))
                conn.commit()
                self._logger.info(f"주문 추가 완료: {len(order_ids)}개의 주문이 저장되었습니다.")
            return order_ids
        except Exception as e:
            self._logger.error(f"주문 추가 중 오류 발생: {str(e)}")
            raise

    def modify_drink(self, order_id: str, drink_type: Optional[str] = None, temperature: Optional[str] = None, size: Optional[str] = None) -> None:
        """특정 주문 수정"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                updates = []
                params = []
                if drink_type:
                    updates.append("drink_type = ?")
                    params.append(drink_type)
                if temperature:
                    updates.append("temperature = ?")
                    params.append(temperature)
                if size:
                    updates.append("size = ?")
                    params.append(size)
                if not updates:
                    raise ValueError("수정할 필드가 없습니다.")
                params.append(order_id)
                cursor.execute(f"UPDATE orders SET {', '.join(updates)} WHERE order_id = ?", params)
                conn.commit()
                self._logger.info(f"주문 {order_id} 수정 완료.")
        except Exception as e:
            self._logger.error(f"주문 수정 중 오류 발생: {str(e)}")
            raise

    def get_drink(self, order_id: str) -> Optional[Drink]:
        """특정 주문 조회"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT drink_type, temperature, size
                    FROM orders WHERE order_id = ?
                ''', (order_id,))
                order_data = cursor.fetchone()
                if not order_data:
                    self._logger.warning(f"주문 {order_id}를 찾을 수 없습니다.")
                    return None
                drink = Drink()
                drink.order_id = order_id
                drink.drink_type, drink.temperature, drink.size = order_data
                cursor.execute('''
                    SELECT option_name FROM order_options WHERE order_id = ?
                ''', (order_id,))
                drink.additional_options = [row[0] for row in cursor.fetchall()]
                return drink
        except Exception as e:
            self._logger.error(f"주문 조회 중 오류 발생: {str(e)}")
            raise

    def delete_order(self, order_id: str) -> None:
        """특정 주문 삭제"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM order_options WHERE order_id = ?', (order_id,))
                cursor.execute('DELETE FROM orders WHERE order_id = ?', (order_id,))
                conn.commit()
                self._logger.info(f"주문 {order_id} 삭제 완료.")
        except Exception as e:
            self._logger.error(f"주문 삭제 중 오류 발생: {str(e)}")
            raise

    def get_order_summary(self) -> List[str]:
        """주문 요약 반환"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT drink_type, temperature, size, COUNT(order_id)
                    FROM orders
                    GROUP BY drink_type, temperature, size
                ''')
                summary = [f"{temp} {drink_type} {size}사이즈 {count}잔" for drink_type, temp, size, count in cursor.fetchall()]
                return summary
        except Exception as e:
            self._logger.error(f"주문 요약 생성 중 오류 발생: {str(e)}")
            raise

    def clear_orders(self) -> None:
        """모든 주문 초기화"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM order_options')
                cursor.execute('DELETE FROM orders')
                conn.commit()
                self._logger.info("모든 주문 초기화 완료.")
        except Exception as e:
            self._logger.error(f"주문 초기화 중 오류 발생: {str(e)}")
            raise


# 저장소 작동 테스트
"""
# 사용 예시
if __name__ == "__main__":
    # 스토리지 초기화
    storage = OrderStorage()
    
    try:
        # 1. 기본 주문 추가 테스트
        drink = Drink()
        drink.drink_type = "아메리카노"
        drink.temperature = "아이스"
        drink.size = "라지"
        drink.quantity = 2
        
        order_ids = storage.add_order(drink)
        print(f"주문 생성 완료 - 주문 ID: {order_ids}")

        # 추가 전 주문 조회 테스트
        print("추가 전")
        for order_id in order_ids:
            drink = storage.get_drink(order_id)
            print(f"\n음료 정보 (ID: {order_id}):")
            print(f"- 종류: {drink.drink_type}")
            print(f"- 온도: {drink.temperature}")
            print(f"- 크기: {drink.size}")
            print(f"- 옵션: {', '.join(drink.additional_options) if drink.additional_options else '없음'}")

        # 5. 주문 요약 테스트
        print("\n현재 주문 내역:")
        for summary in storage.get_order_summary():
            print(f"- {summary}")
        
        # 2. 옵션 추가 테스트
        storage.add_option_to_drink(order_ids[0], "샷", 2)  # 첫번째 음료에 샷 2개 추가
        storage.add_option_to_drink(order_ids[0], "시럽", 1)  # 첫번째 음료에 시럽 1개 추가
        storage.add_option_to_drink(order_ids[1], "휘핑", 1)  # 두번째 음료에 휘핑 1개 추가
        
        # 3. 주문 조회 테스트
        for order_id in order_ids:
            drink = storage.get_drink(order_id)
            print(f"\n음료 정보 (ID: {order_id}):")
            print(f"- 종류: {drink.drink_type}")
            print(f"- 온도: {drink.temperature}")
            print(f"- 크기: {drink.size}")
            print(f"- 옵션: {', '.join(drink.additional_options) if drink.additional_options else '없음'}")

        # 5. 주문 요약 테스트
        print("\n현재 주문 내역:")
        for summary in storage.get_order_summary():
            print(f"- {summary}")
        
        # 4. 옵션 제거 테스트
        storage.remove_option_from_drink(order_ids[0], "샷", 1)  # 첫번째 음료에서 샷 1개 제거
        
        # 5. 주문 요약 테스트
        print("\n현재 주문 내역:")
        for summary in storage.get_order_summary():
            print(f"- {summary}")
        
        # 6. 주문 초기화 테스트
        storage.clear_orders()
        print("\n주문 초기화 후 내역:")
        print("- 주문 없음" if not storage.get_order_summary() else storage.get_order_summary())
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
"""


class ActionOrderConfirmation(Action):
    """음료 주문을 처리하고 저장하는 액션"""

    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()

    def name(self) -> Text:
        """액션 이름"""
        return "action_order_confirmation"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        """Rasa의 액션 실행 메서드"""
        try:
            # 1. 트래커에서 사용자 입력 및 엔티티 추출
            user_text = tracker.latest_message.get("text", "")
            entities = self.mapper.extract_entities(tracker.latest_message)

            # 로깅
            self._logger.info(f"주문 요청 - 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")

            # 2. 음료 객체 생성 및 검증
            user_text = tracker.latest_message.get("text", "")
            drinks = self.mapper.create_drink_from_entities(entities, user_text)

            self.mapper.validate_drink(drinks)

            # 3. 주문 저장
            order_ids = self.storage.add_order(drinks)

            # 4. 주문 요약 생성
            order_summary = self.storage.get_order_summary()

            # 5. 주문 확인 메시지 생성
            drinks_str = "\n".join(f"- {str(drink)}" for drink in drinks)
            summary_str = "\n".join(f"- {item}" for item in order_summary)
            confirmation_message = (
                f"주문이 접수되었습니다.\n"
                f"주문하신 음료:\n{drinks_str}\n"
                f"현재 전체 주문 내역:\n{summary_str}"
            )

            # 6. 메시지 전송
            dispatcher.utter_message(text=confirmation_message)
            self._logger.info(f"주문 확인 완료 - IDs: {order_ids}")
            return []

        except ValueError as e:
            # 사용자 오류 처리
            error_message = f"주문 처리 중 오류가 발생했습니다: {str(e)}"
            self._logger.error(error_message)
            dispatcher.utter_message(text=error_message)
            return []

        except Exception as e:
            # 일반적인 예외 처리
            error_message = "주문 처리 중 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
            return []


# 주문 테스트             

# 주문 테스트 케이스
"""
if __name__ == "__main__":
    storage = OrderStorage()
    storage.clear_orders()
    
    # 테스트 케이스 정의
    test_cases = [
        # 테스트 1: 아아 샷 두번 추가
        {
            'intent': {'name': 'order_coffee', 'confidence': 1.0},
            'entities': [
                {
                    'entity': 'drink_type',
                    'start': 0, 
                    'end': 2,
                    'value': '아아',
                    'extractor': 'RegexEntityExtractor'
                },
                {
                    'entity': 'additional_options',
                    'start': 3,
                    'end': 4,
                    'value': '샷',
                    'extractor': 'RegexEntityExtractor'
                },
                {
                    'entity': 'quantity',
                    'start': 5,
                    'end': 7,
                    'value': '두',
                    'extractor': 'RegexEntityExtractor'
                }
            ],
            'text': '아아 샷 두번 추가해주세요'
        },
        
        # 테스트 2: 아아 샷 두번 추가 두잔
        {
            'intent': {'name': 'order_coffee', 'confidence': 1.0},
            'entities': [
                {
                    'entity': 'drink_type',
                    'start': 0,
                    'end': 2, 
                    'value': '아아',
                    'extractor': 'RegexEntityExtractor'
                },
                {
                    'entity': 'additional_options',
                    'start': 3,
                    'end': 4,
                    'value': '샷',
                    'extractor': 'RegexEntityExtractor'
                },
                {
                    'entity': 'quantity',
                    'start': 5,
                    'end': 7,
                    'value': '두',
                    'extractor': 'RegexEntityExtractor'
                },
                {
                    'entity': 'quantity',
                    'start': 11,
                    'end': 12,
                    'value': '두',
                    'extractor': 'RegexEntityExtractor'
                }
            ],
            'text': '아아 샷 두번 추가 두 잔 주세요'
        },
        
        # 테스트 3: 아메리카노 온도 다르게 두잔
        {
            'intent': {'name': 'order_coffee', 'confidence': 1.0},
            'entities': [
                {
                    'entity': 'drink_type',
                    'start': 0,
                    'end': 5,
                    'value': '아메리카노',
                    'extractor': 'RegexEntityExtractor'
                },
                {
                    'entity': 'quantity',
                    'start': 6,
                    'end': 8,
                    'value': '하나',
                    'extractor': 'RegexEntityExtractor'
                },
                {
                    'entity': 'temperature',
                    'start': 10,
                    'end': 13,
                    'value': '차갑게',
                    'extractor': 'RegexEntityExtractor'
                },
                {
                    'entity': 'quantity',
                    'start': 14,
                    'end': 16,
                    'value': '하나',
                    'extractor': 'RegexEntityExtractor'
                },
                {
                    'entity': 'temperature',
                    'start': 18,
                    'end': 21,
                    'value': '뜨겁게',
                    'extractor': 'RegexEntityExtractor' 
                }
            ],
            'text': '아메리카노 하나는 차갑게 하나는 뜨겁게 주세요'
        }
    ]

    # Action 테스트
    try:
        
        # Action 인스턴스 생성
        action = ActionOrderConfirmation()

        # 테스트용 Tracker와 Dispatcher 목업 
        class MockTracker:
            def __init__(self, message):
                self.latest_message = message

        class MockDispatcher:
            def utter_message(self, text: str):
                print("\n=== 출력 메시지 ===")
                print(text)
                print("================\n")

        # 각 테스트 케이스 실행
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n테스트 케이스 {i} 실행:")
            print(f"입력: {test_case['text']}")
            
            # 저장소 초기화 및 기존 데이터 삭제
            storage = OrderStorage()
            
            # 데이터베이스 파일 삭제 후 재생성
            storage.clear_orders()
            storage._init_database()
            
            tracker = MockTracker(test_case)
            dispatcher = MockDispatcher()
            
            # 비동기 함수 실행
            import asyncio
            asyncio.run(action.run(dispatcher, tracker, {}))
            
            # 현재 주문 상태 출력
            order_summary = storage.get_order_summary()
            print("\n현재 주문 상태:")
            if order_summary:
                for item in order_summary:
                    print(f"- {item}")
            else:
                print("주문 없음")
                
            print("-" * 50)

    except Exception as e:
        print(f"테스트 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
"""        

class ActionModifyOrder(Action):
    """주문 변경 액션"""

    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()

    def name(self) -> Text:
        """액션 이름 반환"""
        return "action_modify_order"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        """Rasa 액션 실행 메서드"""
        try:
            # 1. 트래커에서 사용자 입력 및 엔티티 추출
            user_text = tracker.latest_message.get("text", "")
            entities = tracker.latest_message.get("entities", [])

            self._logger.info(f"주문 변경 요청 - 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")

            # 2. 입력 텍스트를 통해 기존 음료와 새로운 음료 구분
            split_keywords = ["대신", "말고", "은", "는", "을", "를"]
            split_pattern = '|'.join(re.escape(kw) for kw in split_keywords)
            split_text = re.split(f"({split_pattern})", user_text)

            if len(split_text) < 2:
                raise ValueError("변경할 음료와 새로운 음료를 구분할 수 없습니다.")

            # 기존 음료와 새로운 음료의 텍스트 분리
            target_drink_text = split_text[0].strip()
            new_drink_text = split_text[-1].strip()

            # 3. 엔티티를 기준으로 기존 음료와 새로운 음료의 엔티티 분리
            target_entities = [
                entity for entity in entities if target_drink_text in user_text[entity["start"]:entity["end"]]
            ]
            new_entities = [
                entity for entity in entities if new_drink_text in user_text[entity["start"]:entity["end"]]
            ]

            if not target_entities or not new_entities:
                raise ValueError("기존 음료 또는 새로운 음료의 정보를 찾을 수 없습니다.")

            # 4. 음료 객체 생성
            user_text = tracker.latest_message.get("text", "")
            target_drink = self.mapper.create_drink_from_entities(target_entities, user_text)
            new_drink = self.mapper.create_drink_from_entities(new_entities, user_text)


            # 5. 기존 주문 정보 가져오기
            search_conditions = {
                "drink_type": target_drink.drink_type,
                "temperature": target_drink.temperature,
                "size": target_drink.size,
            }
            order_ids = self.storage.get_drink_ids(search_conditions)

            if not order_ids:
                conditions_str = ", ".join(f"{k}: {v}" for k, v in search_conditions.items())
                raise ValueError(f"변경할 주문을 찾을 수 없습니다. 검색 조건: {conditions_str}")

            # 6. 새로운 음료 속성 보완
            if not new_drink.size:
                new_drink.size = target_drink.size
            if not new_drink.temperature:
                new_drink.temperature = target_drink.temperature
            if not new_drink.additional_options:
                new_drink.additional_options = target_drink.additional_options

            self.mapper.validate_drink([new_drink])

            # 7. 수량 처리
            quantity = len(order_ids)
            quantity_to_update = new_drink.quantity if new_drink.quantity else quantity

            if quantity_to_update > quantity:
                # 수량이 더 많은 경우 추가 주문 생성
                additional_needed = quantity_to_update - quantity
                new_order_ids = self.storage.duplicate_order(order_ids[0], additional_needed)
                order_ids.extend(new_order_ids)

            # 8. 주문 변경 수행
            for idx, order_id in enumerate(order_ids[:quantity_to_update]):
                self.storage.modify_drink(
                    order_id,
                    drink_type=new_drink.drink_type,
                    temperature=new_drink.temperature,
                    size=new_drink.size,
                )
                self._logger.debug(f"주문 수정 완료: {order_id}")

                # 옵션 변경
                if new_drink.additional_options:
                    self.storage.clear_options(order_id)
                    for option in new_drink.additional_options:
                        self.storage.add_option_to_drink(order_id, option)

            # 9. 변경된 주문 요약
            order_summary = self.storage.get_order_summary()

            confirmation_message = (
                f"주문이 변경되었습니다.\n"
                f"변경된 음료: {quantity_to_update}잔\n"
                f"현재 전체 주문 내역:\n{chr(10).join(f'- {item}' for item in order_summary)}"
            )

            dispatcher.utter_message(text=confirmation_message)
            self._logger.info(f"주문 변경 완료 - 총 변경된 음료 수: {quantity_to_update}")

            return []

        except ValueError as e:
            error_message = f"주문 변경 중 오류가 발생했습니다: {str(e)}"
            self._logger.error(error_message)
            dispatcher.utter_message(text=error_message)
            return []

        except Exception as e:
            error_message = "주문 변경 중 예상치 못한 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
            return []

# 주문 변경 테스트
"""
# 사용 예시
if __name__ == "__main__":
    
    async def test_order_modification():
        # 스토리지 초기화
        storage = OrderStorage()
        storage.clear_orders()
        
        try:
            # 1. 첫 번째 주문 (아이스 아메리카노 2잔, 샷추가)
            initial_message = {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '아이스',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': "네",
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'size',
                        'value': "라지",
                        'extractor': 'RegexEntityExtractor'
                    }

                ],
                'text': '아이스 아메리카노 네잔  주세요'
            }
            
            # 2. 두 번째 주문 (핫 카페라떼 1잔)
            second_message = {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '핫',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '카페라떼',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '따뜻한 카페라떼 한잔 추가요'
            }
            
            # 3. 주문 변경 요청 (첫 번째 아메리카노를 핫초코로 변경)
            # 테스트용 메시지 수정
            modify_message = {
                'entities': [
                    {'entity': 'drink_type', 'value': '아메리카노', 'start': 0, 'end': 4},
                    {'entity': 'drink_type', 'value': '핫초코', 'start': 7, 'end': 10}, 
                    {'entity': 'quantity', 'value': 5, 'start': 11, 'end': 12},
                    {'entity': 'temperature', 'value': '핫', 'start': 7, 'end': 8}
                ],
                'text': '아메리카노를 핫초코 5개로 변경해주세요'
            }
            
            # Mock 객체 생성
            class MockDispatcher:
                def utter_message(self, text: str):
                    print("\n=== 출력 메시지 ===")
                    print(text)
                    print("================\n")
            
            class MockTracker:
                def __init__(self, message):
                    self.latest_message = message
            
            dispatcher = MockDispatcher()
            
            # 첫 번째 주문 실행
            print("1. 첫 번째 주문 처리 중...")
            order_action = ActionOrderConfirmation()
            await order_action.run(dispatcher, MockTracker(initial_message), {})
            
            # 두 번째 주문 실행
            print("2. 두 번째 주문 처리 중...")
            await order_action.run(dispatcher, MockTracker(second_message), {})
            
            # 주문 변경 실행
            print("3. 주문 변경 처리 중...")
            modify_action = ActionModifyOrder()
            await modify_action.run(dispatcher, MockTracker(modify_message), {})

        except Exception as e:
            print(f"테스트 중 오류 발생: {str(e)}")

    # 비동기 테스트 실행
    import asyncio
    asyncio.run(test_order_modification())
""" 

class ActionSubtractFromOrder(Action):
    """주문 제거 액션"""

    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()

    def name(self) -> Text:
        """액션 이름 반환"""
        return "action_subtract_from_order"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        """Rasa 액션 실행 메서드"""
        try:
            # 1. 사용자 입력 및 엔티티 추출
            user_text = tracker.latest_message.get("text", "")
            entities = tracker.latest_message.get("entities", [])

            self._logger.info(f"주문 제거 요청 - 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")

            # 2. 음료 객체 생성
            drinks = self.mapper.create_drink_from_entities(entities)
            if not drinks or len(drinks) != 1:
                raise ValueError("주문 제거를 위해 하나의 음료만 지정해야 합니다.")
            
            drink_to_remove = drinks[0]

            # 3. 검색 조건 생성
            conditions = {
                "drink_type": drink_to_remove.drink_type,
                "temperature": drink_to_remove.temperature,
                "size": drink_to_remove.size,
            }
            self._logger.debug(f"검색 조건: {conditions}")

            # 조건에 맞는 주문 ID 검색
            order_ids = self.storage.get_drink_ids(conditions)
            if not order_ids:
                raise ValueError(f"조건에 맞는 주문이 없습니다: {conditions}")

            # 4. 제거할 수량 계산
            quantity_to_remove = drink_to_remove.quantity if drink_to_remove.quantity else 1
            if quantity_to_remove > len(order_ids):
                raise ValueError(f"삭제하려는 수량({quantity_to_remove}잔)이 현재 주문 수량({len(order_ids)}잔)을 초과합니다.")

            # 5. 주문 제거
            removed_count = 0
            for order_id in order_ids[:quantity_to_remove]:
                self.storage.delete_order(order_id)
                removed_count += 1
                self._logger.info(f"주문 제거 완료: {order_id} (현재 제거된 수량: {removed_count})")

            # 6. 현재 주문 요약 출력
            order_summary = self.storage.get_order_summary()
            if order_summary:
                confirmation_message = (
                    f"{drink_to_remove.drink_type} {removed_count}잔이 주문에서 제거되었습니다.\n"
                    f"현재 전체 주문 내역:\n{chr(10).join(f'- {item}' for item in order_summary)}"
                )
            else:
                confirmation_message = f"{drink_to_remove.drink_type} {removed_count}잔이 제거되었습니다. 현재 남은 주문이 없습니다."

            dispatcher.utter_message(text=confirmation_message)
            self._logger.info(f"주문 제거 완료 - 제거된 음료 수: {removed_count}")

            return []

        except ValueError as e:
            error_message = f"주문 제거 중 오류가 발생했습니다: {str(e)}"
            self._logger.error(error_message)
            dispatcher.utter_message(text=error_message)
            return []

        except Exception as e:
            error_message = "주문 제거 중 예상치 못한 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
            return []

#주문제거 테스트          
"""
if __name__ == "__main__":
    async def test_subtract_order_scenarios():
        # 스토리지 초기화
        storage = OrderStorage()
        storage.clear_orders()
        
        # 테스트 케이스들
        test_cases = [
            # 1. 초기 주문 생성
            {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '아이스',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '다섯',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아이스 아메리카노 다섯잔 주세요',
                'action': 'order'
            },
            # 2. 두 번째 음료 추가
            {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '따뜻한',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '카페라떼',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '세',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '따뜻한 카페라떼 세잔이요',
                'action': 'order'
            },
            # 3. 첫 번째 음료 일부 제거
            {
                'intent': {'name': 'subtract_order', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '아이스',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '두',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아이스 아메리카노 두잔 빼주세요',
                'action': 'subtract'
            },
            # 4. 존재하지 않는 음료 제거 시도
            {
                'intent': {'name': 'subtract_order', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '에스프레소',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '한',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '에스프레소 한잔 빼주세요',
                'action': 'subtract'
            }
        ]

        class MockDispatcher:
            def utter_message(self, text: str):
                print("\n=== 출력 메시지 ===")
                print(text)
                print("================\n")

        class MockTracker:
            def __init__(self, message):
                self.latest_message = message

        dispatcher = MockDispatcher()

        try:
            # 각 테스트 케이스 실행
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n테스트 케이스 {i} 실행 중...")
                print(f"의도: {test_case['intent']['name']}")
                print(f"텍스트: {test_case['text']}")
                
                tracker = MockTracker(test_case)
                
                if test_case['action'] == 'order':
                    action = ActionOrderConfirmation()
                elif test_case['action'] == 'subtract':
                    action = ActionSubtractFromOrder()
                
                await action.run(dispatcher, tracker, {})
                
                # 현재 주문 상태 출력
                order_summary = storage.get_order_summary()
                print("\n현재 주문 상태:")
                for item in order_summary:
                    print(f"- {item}")
                print(f"\n테스트 케이스 {i} 완료\n")

        except Exception as e:
            print(f"테스트 중 오류 발생: {str(e)}")

    # 비동기 테스트 실행
    import asyncio
    asyncio.run(test_subtract_order_scenarios())
"""

# 주문 수량 변경
class ActionAddSubtract(Action):
    """주문 수량 추가/삭제 액션"""

    def __init__(self):
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()
        self._logger = LoggerSetup.get_logger()

    def name(self) -> Text:
        """액션 이름 반환"""
        return "action_add_subtract"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 1. 사용자 입력 및 엔티티 추출
            user_text = tracker.latest_message.get("text", "")
            entities = tracker.latest_message.get("entities", [])

            self._logger.info(f"주문 수량 변경 요청 - 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")

            # 2. 음료 객체 생성
            drinks = self.mapper.create_drink_from_entities(entities)
            if not drinks or len(drinks) != 1:
                raise ValueError("수량 변경을 위해 하나의 음료만 지정해야 합니다.")
            
            drink_to_modify = drinks[0]

            # 3. 검색 조건 생성
            search_conditions = {
                "drink_type": drink_to_modify.drink_type,
                "temperature": drink_to_modify.temperature,
                "size": drink_to_modify.size,
            }
            self._logger.debug(f"검색 조건: {search_conditions}")

            # 조건에 맞는 주문 ID 검색
            order_ids = self.storage.get_drink_ids(search_conditions)
            if not order_ids:
                conditions_str = ", ".join(f"{k}: {v}" for k, v in search_conditions.items())
                raise ValueError(f"조건에 맞는 주문을 찾을 수 없습니다. 검색 조건: {conditions_str}")

            # 4. 수량 변경 처리
            quantity_change = drink_to_modify.quantity if drink_to_modify.quantity else 1
            current_quantity = len(order_ids)

            if "추가" in user_text or "더" in user_text:
                # 주문 추가
                new_ids = self.storage.duplicate_order(order_ids[0], quantity_change)
                modified_count = len(new_ids)
                action_type = "추가"
            elif "제거" in user_text or "삭제" in user_text or "빼" in user_text:
                # 주문 삭제
                if quantity_change > current_quantity:
                    raise ValueError(f"현재 주문 수량({current_quantity}잔)보다 많은 수량({quantity_change}잔)을 삭제할 수 없습니다.")
                
                for order_id in order_ids[:quantity_change]:
                    self.storage.delete_order(order_id)
                modified_count = quantity_change
                action_type = "삭제"
            else:
                raise ValueError("명확한 액션(추가/삭제)이 지정되지 않았습니다.")

            # 5. 주문 요약 조회
            order_summary = self.storage.get_order_summary()

            # 6. 메시지 생성 및 전송
            confirmation_message = (
                f"주문이 {action_type}되었습니다.\n"
                f"{action_type}된 음료: {modified_count}잔\n"
                f"현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            dispatcher.utter_message(text=confirmation_message)
            self._logger.info(f"주문 {action_type} 완료 - {action_type}된 음료 수: {modified_count}")

            return []

        except ValueError as e:
            error_message = f"주문 수량 변경 중 오류가 발생했습니다: {str(e)}"
            self._logger.error(error_message)
            dispatcher.utter_message(text=error_message)
            return []

        except Exception as e:
            error_message = "주문 수량 변경 중 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
            return []

# 주문 수량변경 테스트
"""
# 테스트 코드
if __name__ == "__main__":
    async def test_add_subtract_scenarios():
        # 스토리지 초기화
        storage = OrderStorage()
        storage.clear_orders()
        print("\n1. 첫 번째 주문 처리 중...")
        
        try:
            # 1. 초기 주문 생성 (아이스 아메리카노 4잔)
            initial_message = {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '아이스',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '네',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'size',
                        'value': '라지',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아이스 아메리카노 네잔  주세요'
            }

            # 2. 주문 추가 메시지 (아메리카노 2잔 추가)
            add_message = {
                'intent': {'name': 'add_order', 'confidence': 0.98},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '두',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아메리카노 두잔 더 추가해주세요'
            }

            # 3. 주문 감소 메시지 (아이스 아메리카노 3잔 빼기)
            subtract_message = {
                'intent': {'name': 'subtract_order', 'confidence': 0.95},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '아이스',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '세',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아이스 아메리카노 세잔 빼주세요'
            }

            class MockDispatcher:
                def utter_message(self, text: str):
                    print("\n=== 출력 메시지 ===")
                    print(text)
                    print("================\n")

            class MockTracker:
                def __init__(self, message):
                    self.latest_message = message

            dispatcher = MockDispatcher()

            # 초기 주문 생성
            print("1. 첫 번째 주문 처리 중...")
            initial_action = ActionOrderConfirmation()
            await initial_action.run(dispatcher, MockTracker(initial_message), {})
            print("2. 두 번째 주문 처리 중...")

            # 주문 추가 테스트
            add_action = ActionAddSubtract()
            await add_action.run(dispatcher, MockTracker(add_message), {})
            print("3. 주문 변경 처리 중...")

            # 주문 감소 테스트
            subtract_action = ActionAddSubtract()
            await subtract_action.run(dispatcher, MockTracker(subtract_message), {})

        except Exception as e:
            print(f"테스트 중 오류 발생: {str(e)}")

    # 비동기 테스트 실행
    import asyncio
    asyncio.run(test_add_subtract_scenarios())
"""
# 주문 확인
class ActionOrderFinish(Action):
    """주문 확인 액션"""

    def __init__(self):
        self.storage = OrderStorage()
        self._logger = LoggerSetup.get_logger()

    def name(self) -> Text:
        return "action_order_finish"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 현재 주문 내역 요약 조회
            order_summary = self.storage.get_order_summary()

            if not order_summary:
                dispatcher.utter_message(text="현재 주문이 없습니다.")
                return []

            # 주문 요약 메시지 생성
            confirmation_message = (
                f"현재 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}\n"
                f"주문을 완료하시겠습니까?"
            )

            # 메시지 전송
            dispatcher.utter_message(text=confirmation_message)
            self._logger.info("주문 확인 메시지 전송 완료")
            return []

        except Exception as e:
            error_message = "주문 확인 중 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"주문 확인 중 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
            return []

# 주문 확인 테스트
"""
if __name__ == "__main__":
    async def test_order_finish_scenarios():
        # 스토리지 초기화
        storage = OrderStorage()
        storage.clear_orders()
        print("\n주문 완료 테스트 시작...")
        
        # 테스트 케이스들
        test_cases = [
            # 1. 첫 번째 주문 생성
            {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '아이스',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '두',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'size',
                        'value': '라지',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아이스 아메리카노 라지 두잔 주세요',
                'action': 'order'
            },
            # 2. 두 번째 주문 추가
            {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '따뜻한',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '카페라떼',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'option',
                        'value': '샷추가',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '따뜻한 카페라떼 샷추가로 주세요',
                'action': 'order'
            },
            # 3. 주문 확인
            {
                'intent': {'name': 'check_order', 'confidence': 1.0},
                'entities': [],
                'text': '주문 확인해주세요',
                'action': 'finish'
            },
            # 4. 빈 주문 상태에서 확인
            {
                'intent': {'name': 'check_order', 'confidence': 1.0},
                'entities': [],
                'text': '주문 확인해주세요',
                'action': 'finish',
                'clear_before': True  # 이 케이스 전에 주문 초기화
            }
        ]

        class MockDispatcher:
            def utter_message(self, text: str):
                print("\n=== 출력 메시지 ===")
                print(text)
                print("================\n")

        class MockTracker:
            def __init__(self, message):
                self.latest_message = message

        dispatcher = MockDispatcher()

        try:
            # 각 테스트 케이스 실행
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n테스트 케이스 {i} 실행 중...")
                print(f"의도: {test_case['intent']['name']}")
                print(f"텍스트: {test_case['text']}")
                
                # 필요한 경우 주문 초기화
                if test_case.get('clear_before', False):
                    print("\n주문 초기화 수행...")
                    storage.clear_orders()
                
                tracker = MockTracker(test_case)
                
                if test_case['action'] == 'order':
                    action = ActionOrderConfirmation()
                elif test_case['action'] == 'finish':
                    action = ActionOrderFinish()
                
                await action.run(dispatcher, tracker, {})
                
                # 현재 주문 상태 출력
                order_summary = storage.get_order_summary()
                print("\n현재 주문 상태:")
                if order_summary:
                    for item in order_summary:
                        print(f"- {item}")
                else:
                    print("주문 없음")
                    
                print(f"\n테스트 케이스 {i} 완료\n")
                print("-" * 50)

        except Exception as e:
            print(f"테스트 중 오류 발생: {str(e)}")
            print(f"오류 상세: {type(e).__name__}")
            import traceback
            print(traceback.format_exc())

    # 비동기 테스트 실행
    import asyncio
    asyncio.run(test_order_finish_scenarios())
"""      
# 주문 취소 (초기화)
class ActionCancelOrder(Action):
    """주문 취소 액션"""

    def __init__(self):
        self.storage = OrderStorage()
        self._logger = LoggerSetup.get_logger()

    def name(self) -> Text:
        return "action_cancel_order"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 모든 주문 초기화
            self.storage.clear_orders()

            # 사용자에게 초기화 완료 메시지 전달
            dispatcher.utter_message(text="모든 주문이 초기화되었습니다.")
            self._logger.info("주문 초기화 완료")
            return []

        except Exception as e:
            error_message = "주문 초기화 중 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"주문 초기화 중 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
            return []

# 주문 초기화 테스트
"""
if __name__ == "__main__":
    async def test_cancel_order_scenarios():
        # 스토리지 초기화
        storage = OrderStorage()
        storage.clear_orders()
        
        # 테스트 케이스들
        test_cases = [
            # 1. 첫 번째 주문 생성
            {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '아이스',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '두',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아이스 아메리카노 두잔 주세요',
                'action': 'order'
            },
            # 2. 두 번째 주문 추가
            {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '따뜻한',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '카페라떼',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'size',
                        'value': '라지',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '따뜻한 카페라떼 라지 사이즈로 주세요',
                'action': 'order'
            },
            # 3. 주문 취소
            {
                'intent': {'name': 'cancel_order', 'confidence': 1.0},
                'entities': [],
                'text': '주문 전체 취소해주세요',
                'action': 'cancel'
            },
            # 4. 취소 후 새로운 주문
            {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '에스프레소',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '한',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '에스프레소 한잔이요',
                'action': 'order'
            }
        ]

        class MockDispatcher:
            def utter_message(self, text: str):
                print("\n=== 출력 메시지 ===")
                print(text)
                print("================\n")

        class MockTracker:
            def __init__(self, message):
                self.latest_message = message

        dispatcher = MockDispatcher()

        try:
            print("\n주문 취소 테스트 시작...")
            
            # 각 테스트 케이스 실행
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n테스트 케이스 {i} 실행 중...")
                print(f"의도: {test_case['intent']['name']}")
                print(f"텍스트: {test_case['text']}")
                
                tracker = MockTracker(test_case)
                
                if test_case['action'] == 'order':
                    action = ActionOrderConfirmation()
                elif test_case['action'] == 'cancel':
                    action = ActionCancelOrder()
                
                await action.run(dispatcher, tracker, {})
                
                # 현재 주문 상태 출력
                order_summary = storage.get_order_summary()
                print("\n현재 주문 상태:")
                if order_summary:
                    for item in order_summary:
                        print(f"- {item}")
                else:
                    print("주문 없음")
                    
                print(f"\n테스트 케이스 {i} 완료\n")
                print("-" * 50)

        except Exception as e:
            print(f"테스트 중 오류 발생: {str(e)}")

    # 비동기 테스트 실행
    import asyncio
    asyncio.run(test_cancel_order_scenarios())
"""
    # 커피 사이즈 변경
class ActionSelectCoffeeSize(Action):
    """음료 사이즈 변경 액션"""

    def __init__(self):
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()
        self._logger = LoggerSetup.get_logger()

    def name(self) -> Text:
        return "action_select_coffee_size"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 1. 사용자 입력에서 엔티티 추출
            entities = tracker.latest_message.get("entities", [])
            user_text = tracker.latest_message.get("text", "")
            self._logger.info(f"(사이즈 변경 요청) 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")

            # 2. 음료 객체 생성
            drinks = self.mapper.create_drink_from_entities(entities)
            if not drinks or len(drinks) != 1:
                raise ValueError("사이즈 변경을 위해 하나의 음료만 지정해야 합니다.")

            drink_to_modify = drinks[0]
            if not drink_to_modify.size:
                raise ValueError("변경할 사이즈가 명시되지 않았습니다.")

            # 3. 조건에 맞는 주문 검색
            search_conditions = {
                "drink_type": drink_to_modify.drink_type,
                "temperature": drink_to_modify.temperature,
                "size": drink_to_modify.size,
            }
            order_ids = self.storage.get_drink_ids(search_conditions)
            if not order_ids:
                conditions_str = ", ".join(f"{k}: {v}" for k, v in search_conditions.items())
                raise ValueError(f"해당하는 주문({conditions_str})을 찾을 수 없습니다.")

            # 4. 사이즈 변경 처리
            modified_drinks = []
            for order_id in order_ids:
                self.storage.modify_drink(
                    order_id,
                    drink_type=drink_to_modify.drink_type,
                    temperature=drink_to_modify.temperature,
                    size=drink_to_modify.size,
                )
                modified_drinks.append(order_id)

            # 5. 변경 완료 메시지 생성
            order_summary = self.storage.get_order_summary()
            confirmation_message = (
                f"총 {len(modified_drinks)}개의 음료 사이즈가 '{drink_to_modify.size}'로 변경되었습니다.\n"
                f"현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            dispatcher.utter_message(text=confirmation_message)
            self._logger.info(f"사이즈 변경 완료 - 변경된 음료 수: {len(modified_drinks)}")
            return []

        except ValueError as e:
            error_message = f"사이즈 변경 중 오류 발생: {str(e)}"
            self._logger.error(error_message)
            dispatcher.utter_message(text=error_message)
            return []
        except Exception as e:
            error_message = "사이즈 변경 중 예상치 못한 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
            return []

class ActionSelectCoffeeTemperature(Action):
    """음료 온도 변경 액션"""

    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()

    def name(self) -> Text:
        return "action_select_coffee_temperature"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 사용자 입력과 엔티티 추출
            entities = tracker.latest_message.get("entities", [])
            user_text = tracker.latest_message.get("text", "")
            self._logger.info(f"(온도 변경 요청) 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")

            # 음료 객체 생성 및 온도 설정
            user_text = tracker.latest_message.get("text", "")
            drink = self.mapper.create_drink_from_entities(entities, user_text)

            self._logger.info(f"(온도 변경) 표준화된 음료: {drink}")

            if not drink.drink_type:
                raise ValueError("변경할 음료 종류가 입력되지 않았습니다.")

            if not drink.temperature:
                # 현재 음료의 마지막 주문 상태에서 반대 온도로 변경
                drink_ids = self.storage.get_drink_ids({"drink_type": drink.drink_type})
                if not drink_ids:
                    raise ValueError(f"'{drink.drink_type}'에 대한 주문이 없습니다.")
                current_drink = self.storage.get_drink(drink_ids[-1])
                drink.temperature = "핫" if current_drink.temperature == "아이스" else "아이스"

            # 해당 음료 검색 및 온도 변경
            order_ids = self.storage.get_drink_ids({"drink_type": drink.drink_type})
            if not order_ids:
                raise ValueError(f"'{drink.drink_type}'에 대한 주문을 찾을 수 없습니다.")

            for order_id in order_ids:
                self.storage.modify_drink(order_id, temperature=drink.temperature)
                self._logger.info(f"온도가 '{drink.temperature}'로 변경되었습니다 - 주문 ID: {order_id}")

            # 주문 요약
            order_summary = self.storage.get_order_summary()
            confirmation_message = (
                f"'{drink.drink_type}' 음료의 온도가 '{drink.temperature}'로 변경되었습니다.\n"
                f"현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            dispatcher.utter_message(text=confirmation_message)

            return []

        except ValueError as e:
            self._logger.error(f"온도 변경 중 오류 발생: {str(e)}")
            dispatcher.utter_message(text=f"온도 변경 중 오류가 발생했습니다: {str(e)}")
            return []
        except Exception as e:
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text="온도 변경 중 오류가 발생했습니다. 다시 시도해주세요.")
            return []


"""
#사이즈, 온도 변경 테스트
if __name__ == "__main__":
    async def test_coffee_modification_scenarios():
        # 스토리지 초기화
        storage = OrderStorage()
        storage.clear_orders()
        print("\n커피 속성 변경 테스트 시작...")
        
        # 테스트 케이스들
        test_cases = [
            # 1. 초기 주문 생성
            {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '아이스',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '두',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아이스 아메리카노 두잔 주세요',
                'action': 'order'
            },
            # 2. 사이즈 변경 요청
            {
                'intent': {'name': 'change_size', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'size',
                        'value': '라지',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아메리카노 라지 사이즈로 변경해주세요',
                'action': 'change_size'
            },
            # 3. 온도 변경 요청
            {
                'intent': {'name': 'change_temperature', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'temperature',
                        'value': '따뜻한',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아메리카노 따뜻하게 변경해주세요',
                'action': 'change_temperature'
            },
            # 4. 존재하지 않는 음료 변경 시도
            {
                'intent': {'name': 'change_size', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '카페라떼',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'size',
                        'value': '라지',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '카페라떼 라지 사이즈로 변경해주세요',
                'action': 'change_size'
            },
            # 5. 속성 없이 변경 시도
            {
                'intent': {'name': 'change_temperature', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아메리카노 온도 변경해주세요',
                'action': 'change_temperature'
            }
        ]

        class MockDispatcher:
            def utter_message(self, text: str):
                print("\n=== 출력 메시지 ===")
                print(text)
                print("================\n")

        class MockTracker:
            def __init__(self, message):
                self.latest_message = message

        dispatcher = MockDispatcher()

        try:
            # 각 테스트 케이스 실행
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n테스트 케이스 {i} 실행 중...")
                print(f"의도: {test_case['intent']['name']}")
                print(f"텍스트: {test_case['text']}")
                
                tracker = MockTracker(test_case)
                
                if test_case['action'] == 'order':
                    action = ActionOrderConfirmation()
                elif test_case['action'] == 'change_size':
                    action = ActionSelectCoffeeSize()
                elif test_case['action'] == 'change_temperature':
                    action = ActionSelectCoffeeTemperature()
                
                await action.run(dispatcher, tracker, {})
                
                # 현재 주문 상태 출력
                order_summary = storage.get_order_summary()
                print("\n현재 주문 상태:")
                if order_summary:
                    for item in order_summary:
                        print(f"- {item}")
                else:
                    print("주문 없음")
                    
                print(f"\n테스트 케이스 {i} 완료\n")
                print("-" * 50)

        except Exception as e:
            print(f"테스트 중 오류 발생: {str(e)}")
            print(f"오류 상세: {type(e).__name__}")
            import traceback
            print(traceback.format_exc())

    # 비동기 테스트 실행
    import asyncio
    asyncio.run(test_coffee_modification_scenarios())
"""

# 커피 추가옵션 추가
class ActionAddAdditionalOption(Action):
    """음료에 추가 옵션 추가"""

    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()

    def name(self) -> Text:
        return "action_add_additional_option"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 사용자 입력 및 엔티티 추출
            entities = tracker.latest_message.get("entities", [])
            user_text = tracker.latest_message.get("text", "")
            self._logger.info(f"(추가 옵션 추가 요청) 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")

            # 음료 객체 생성
            drink = self.mapper.create_drink_from_entities(entities)
            if not drink.additional_options:
                raise ValueError("추가할 옵션이 명시되지 않았습니다.")
            if not drink.drink_type:
                raise ValueError("음료 종류가 입력되지 않았습니다.")

            # 옵션 추가
            order_ids = self.storage.get_drink_ids({"drink_type": drink.drink_type})
            if not order_ids:
                raise ValueError(f"'{drink.drink_type}'에 대한 주문을 찾을 수 없습니다.")

            for option in drink.additional_options:
                self.storage.add_option_to_drink(order_ids[-1], option)
                self._logger.info(f"'{option}' 옵션이 추가되었습니다 - 주문 ID: {order_ids[-1]}")

            # 주문 요약
            order_summary = self.storage.get_order_summary()
            confirmation_message = (
                f"'{drink.drink_type}' 음료에 옵션 '{', '.join(drink.additional_options)}'이 추가되었습니다.\n"
                f"현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            dispatcher.utter_message(text=confirmation_message)

            return []

        except ValueError as e:
            self._logger.error(f"옵션 추가 중 오류 발생: {str(e)}")
            dispatcher.utter_message(text=f"옵션 추가 중 오류가 발생했습니다: {str(e)}")
            return []
        except Exception as e:
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text="옵션 추가 중 오류가 발생했습니다. 다시 시도해주세요.")
            return []


class ActionRemoveAdditionalOption(Action):
    """음료에서 추가 옵션 제거"""

    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()

    def name(self) -> Text:
        return "action_remove_additional_option"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 사용자 입력 및 엔티티 추출
            entities = tracker.latest_message.get("entities", [])
            user_text = tracker.latest_message.get("text", "")
            self._logger.info(f"(추가 옵션 제거 요청) 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")

            # 음료 객체 생성
            user_text = tracker.latest_message.get("text", "")
            drink = self.mapper.create_drink_from_entities(entities, user_text)

            if not drink.additional_options:
                raise ValueError("제거할 옵션이 명시되지 않았습니다.")
            if not drink.drink_type:
                raise ValueError("음료 종류가 입력되지 않았습니다.")

            # 옵션 제거
            order_ids = self.storage.get_drink_ids({"drink_type": drink.drink_type})
            if not order_ids:
                raise ValueError(f"'{drink.drink_type}'에 대한 주문을 찾을 수 없습니다.")

            for option in drink.additional_options:
                self.storage.remove_option_from_drink(order_ids[-1], option)
                self._logger.info(f"'{option}' 옵션이 제거되었습니다 - 주문 ID: {order_ids[-1]}")

            # 주문 요약
            order_summary = self.storage.get_order_summary()
            confirmation_message = (
                f"'{drink.drink_type}' 음료에서 옵션 '{', '.join(drink.additional_options)}'이 제거되었습니다.\n"
                f"현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            dispatcher.utter_message(text=confirmation_message)

            return []

        except ValueError as e:
            self._logger.error(f"옵션 제거 중 오류 발생: {str(e)}")
            dispatcher.utter_message(text=f"옵션 제거 중 오류가 발생했습니다: {str(e)}")
            return []
        except Exception as e:
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text="옵션 제거 중 오류가 발생했습니다. 다시 시도해주세요.")
            return []


""" 
if __name__ == "__main__":
    async def test_remove_option_scenarios():
        # 스토리지 초기화
        storage = OrderStorage()
        storage.clear_orders()
        print("\n추가 옵션 제거 테스트 시작...")
        
        # 테스트 케이스들
        test_cases = [
            # 1. 초기 주문 생성 (샷 추가 2번)
            {
                'intent': {'name': 'order_coffee', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'temperature',
                        'value': '아이스',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'additional_options',
                        'value': '샷',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아이스 아메리카노에 샷 추가해주세요',
                'action': 'order'
            },
            # 2. 샷 1개 제거 요청
            {
                'intent': {'name': 'remove_option', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'additional_options',
                        'value': '샷',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'quantity',
                        'value': '한',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아메리카노 샷 하나만 빼주세요',
                'action': 'remove_option'
            },
            # 3. 모든 샷 제거 요청
            {
                'intent': {'name': 'remove_option', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'additional_options',
                        'value': '샷',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아메리카노 샷 다 빼주세요',
                'action': 'remove_option'
            },
            # 4. 존재하지 않는 옵션 제거 시도
            {
                'intent': {'name': 'remove_option', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '아메리카노',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'additional_options',
                        'value': '시럽',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '아메리카노 시럽 빼주세요',
                'action': 'remove_option'
            },
            # 5. 존재하지 않는 음료의 옵션 제거 시도
            {
                'intent': {'name': 'remove_option', 'confidence': 1.0},
                'entities': [
                    {
                        'entity': 'drink_type',
                        'value': '카페라떼',
                        'extractor': 'RegexEntityExtractor'
                    },
                    {
                        'entity': 'additional_options',
                        'value': '샷',
                        'extractor': 'RegexEntityExtractor'
                    }
                ],
                'text': '카페라떼 샷 빼주세요',
                'action': 'remove_option'
            }
        ]

        class MockDispatcher:
            def utter_message(self, text: str):
                print("\n=== 출력 메시지 ===")
                print(text)
                print("================\n")

        class MockTracker:
            def __init__(self, message):
                self.latest_message = message

        dispatcher = MockDispatcher()

        try:
            # 각 테스트 케이스 실행
            for i, test_case in enumerate(test_cases, 1):
                print(f"\n테스트 케이스 {i} 실행 중...")
                print(f"의도: {test_case['intent']['name']}")
                print(f"텍스트: {test_case['text']}")
                
                tracker = MockTracker(test_case)
                
                if test_case['action'] == 'order':
                    action = ActionOrderConfirmation()
                elif test_case['action'] == 'remove_option':
                    action = ActionRemoveAdditionalOption()
                
                await action.run(dispatcher, tracker, {})
                
                # 현재 주문 상태 출력
                order_summary = storage.get_order_summary()
                print("\n현재 주문 상태:")
                if order_summary:
                    for item in order_summary:
                        print(f"- {item}")
                else:
                    print("주문 없음")
                    
                print(f"\n테스트 케이스 {i} 완료\n")
                print("-" * 50)

        except Exception as e:
            print(f"테스트 중 오류 발생: {str(e)}")
            print(f"오류 상세: {type(e).__name__}")
            import traceback
            print(traceback.format_exc())

    # 비동기 테스트 실행
    import asyncio
    asyncio.run(test_remove_option_scenarios())
"""
#########################
                      
#아래부터는 필요없음     
                       
#########################


# 주문 테이크아웃 판별
class ActionTakeOut(Action):
    def name(self) -> Text:
        return "action_takeout"

    # 주문을 완료하는 액션 실행
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:          
            # 최신 사용자 메시지에서 DIETClassifier가 아닌 엔티티를 가져오기
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]

            logging.warning(f"테이크아웃 엔티티: {entities}")

            sorted_entities = sorted(entities, key=lambda e: e.get("start", 0))
            take_entities = [entity for entity in sorted_entities if entity.get("entity") == "take"]

            if take_entities:
                last_take_value = take_entities[-1].get("value")
                takeout = last_take_value 

                final_message = f"{takeout} 주문이 완료되었습니다. 결제는 하단의 카드리더기로 결제해 주시기 바랍니다. 감사합니다."
                dispatcher.utter_message(text=final_message)
            else:
                dispatcher.utter_message(text="테이크아웃 여부를 확인할 수 없습니다.")

            return []
        except Exception as e:
            logging.exception("Exception occurred in action_takeout")
            dispatcher.utter_message(text="주문 완료 중 오류가 발생했습니다. 다시 시도해주세요.")
            return []

# 커피 추천
class ActionCoffeeRecommendation(Action):
    def name(self) -> Text:
        return "action_coffee_recommendation"
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        recommended_coffees = ["아메리카노", "카페라떼", "카푸치노", "에스프레소"]
        recommended_coffees_str = ", ".join(recommended_coffees)
        recommedded_message = f"저희 매장이 추천하는 커피로는 {recommended_coffees_str} 등이 있습니다. 어떤 커피을 원하시나요?"
        dispatcher.utter_message(text=recommedded_message)

        return []
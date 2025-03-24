import sqlite3
import json
from typing import Any, Text, Dict, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import logging
from datetime import datetime
import re

# 로거 설정
class LoggerSetup:
    _logger = None

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            cls._logger = cls.setup_logger()
        return cls._logger

    @staticmethod
    def setup_logger():
        """로깅 설정을 초기화하고 logger 인스턴스를 반환함"""
        
        # 로거 생성
        logger = logging.getLogger('CoffeeBot')
        logger.setLevel(logging.DEBUG)

        # 이미 핸들러가 있다면 제거 (중복 방지)
        if logger.handlers:
            logger.handlers.clear()

        # 콘솔 출력 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 로그 포맷 설정
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # 핸들러 추가
        logger.addHandler(console_handler)

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
            "한": 1, "하나": 1, "1": 1,
            "두": 2, "둘": 2, "2": 2, 
            "세": 3, "셋": 3, "3": 3,
            "네": 4, "넷": 4, "4": 4,
            "다섯": 5, "5": 5,
            "여섯": 6, "6": 6,
            "일곱": 7, "7": 7,
            "여덟": 8, "8": 8,
            "아홉": 9, "9": 9,
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
        
# 음료 객체 생성 매서드
class Drink:
    """음료 객체를 표현하는 클래스"""
    
    def __init__(self):
        self.drink_type: str = None
        self.temperature: str = "핫"  # 기본값
        self.size: str = "미디움"     # 기본값
        self.quantity: int = 1       # 기본값
        self.additional_options: List[str] = []
        self.order_id: str = None

    def __str__(self) -> str:
        if self.additional_options:
            return f"{self.temperature} {self.drink_type} {self.size}사이즈 {', '.join(self.additional_options)}추가 {self.quantity}잔"
        else:
            return f"{self.temperature} {self.drink_type} {self.size}사이즈 {self.quantity}잔"

# 음료 객제 매핑
class DrinkMapper:
    """RASA 엔티티를 음료 객체로 매핑하는 클래스"""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger()
        self.standardizer = DrinkStandardizer()
    
    def extract_entities(self, tracker_message: Dict) -> List[Dict]:
        """RASA 메시지에서 유효한 엔티티만 추출"""
        entities = tracker_message.get("entities", [])
        # DIETClassifier가 아닌 엔티티만 필터링
        valid_entities = [
            entity for entity in entities 
            if entity.get("extractor") != "DIETClassifier"
        ]
        self.logger.debug(f"추출된 엔티티: {valid_entities}")
        return valid_entities

    def create_drink_from_entities(self, entities: List[Dict]) -> Drink:
        """엔티티 리스트로부터 음료 객체 생성"""
        drink = Drink()
        
        try:
            # 시작 위치를 기준으로 엔티티 정렬
            
            sorted_entities = sorted(entities, key=lambda x: x.get("start", 0))
            user_text = tracker.latest_message.get("text", "")
            for i, entity in enumerate(sorted_entities):
                entity_type = entity.get("entity")
                value = entity.get("value")
                
                if not entity_type or not value:
                    continue
                
                # 값 표준화
                standardized_value = DrinkStandardizer.standardize(entity_type, value)
                
                # 엔티티 타입에 따라 속성 설정
                if entity_type == "quantity":
                    # 이전 엔티티가 additional_options인 경우 옵션 수량 처리
                    if i > 0 and sorted_entities[i-1]["entity"] == "additional_options":
                        next_word = user_text[entity["end"]:].strip().split()[0] if entity["end"] < len(user_text) else ""
                        
                        # "번", "개"로 끝나는 경우 옵션 수량으로 처리
                        if next_word in ["번", "개", "번추가", "개추가"]:
                            option = sorted_entities[i-1]["value"]
                            option_count = int(standardized_value)
                            # 기존 옵션 제거 후 수량만큼 추가
                            if option in drink.additional_options:
                                drink.additional_options.remove(option)
                            for _ in range(option_count):
                                drink.additional_options.append(option)
                            continue
                    
                    # 일반 음료 수량 처리
                    if isinstance(standardized_value, int):
                        drink.quantity = standardized_value
                    else:
                        try:
                            drink.quantity = int(standardized_value)
                        except ValueError:
                            self.logger.warning(f"잘못된 수량 값: {standardized_value}, 기본값 1 사용")
                            drink.quantity = 1
                elif entity_type == 'drink_type':
                    if value in ["아아", "아 아", "아", "아가"]:
                        drink.drink_type = "아메리카노"
                        drink.temperature = "아이스"
                    elif value in ["뜨아", "뜨 아", "뜨아아", "또", "응아", "쁘허", "뚜아"]:
                        drink.drink_type = "아메리카노"
                        drink.temperature = "핫"
                    elif value in ["아카라", "아까라"]:
                        drink.temperature = "아이스"
                        drink.drink_type = "카페라떼"
                    elif value in ["아샷추", "아샤추", "아샷 추", "아샸츄","아사츄","아샤츄", "아사추"]:
                        drink.additional_options = ["샷"]
                        drink.drink_type = "복숭아아이스티"
                    else:
                        drink.drink_type = standardized_value
                elif entity_type == "temperature":
                    drink.temperature = standardized_value
                elif entity_type == "size":
                    drink.size = standardized_value
                elif entity_type == "additional_options":
                    drink.additional_options.append(standardized_value)
            
            self.logger.info(f"음료 객체 생성 완료: {str(drink)}")
            return drink
            
        except Exception as e:
            self.logger.error(f"음료 객체 생성 중 오류 발생: {str(e)}")
            raise ValueError(f"음료 객체 생성 실패: {str(e)}")

    def validate_drink(self, drink: Drink) -> bool:
        """음료 객체의 유효성 검사"""
        try:
            # 필수 속성 검사
            if not drink.drink_type:
                raise ValueError("음료 종류가 지정되지 않았습니다")
            
            # 온도 제약조건 검사
            if drink.drink_type in DrinkStandardizer._drink_constraints["ice_only"] and drink.temperature != "아이스":
                raise ValueError(f"{drink.drink_type}는 아이스로만 주문 가능합니다")
            
            if drink.drink_type in DrinkStandardizer._drink_constraints["hot_only"] and drink.temperature != "핫":
                raise ValueError(f"{drink.drink_type}는 핫으로만 주문 가능합니다")
            
            self.logger.info(f"음료 유효성 검사 통과: {str(drink)}")
            return True
            
        except Exception as e:
            self.logger.error(f"음료 유효성 검사 실패: {str(e)}")
            raise

#스토리지 관리 매서드
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
                
                # 옵션 테이블 생성 (다중 옵션 저장)
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS order_options (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT,
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

    def add_order(self, drink: Drink) -> List[str]:
        """음료 주문을 개별 주문으로 분리하여 저장"""
        try:
            order_ids = []
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                
                # quantity 만큼 개별 주문 생성
                for _ in range(drink.quantity):
                    order_id = self._generate_order_id()
                    order_ids.append(order_id)
                    
                    # 주문 기본 정보 저장
                    cursor.execute('''
                        INSERT INTO orders (order_id, drink_type, temperature, size)
                        VALUES (?, ?, ?, ?)
                    ''', (order_id, drink.drink_type, drink.temperature, drink.size))
                    
                    # 옵션 저장
                    for option in drink.additional_options:
                        cursor.execute('''
                            INSERT INTO order_options (order_id, option_name)
                            VALUES (?, ?)
                        ''', (order_id, option))
                
                conn.commit()
                self._logger.info(f"주문 추가 완료: {drink.drink_type} {drink.quantity}잔")
                return order_ids
                
        except Exception as e:
            self._logger.error(f"주문 추가 중 오류 발생: {str(e)}")
            raise
    
    def modify_drink(self, order_id: str, drink_type: Optional[str] = None, temperature: Optional[str] = None, size: Optional[str] = None) -> None:
        """주문을 수정"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                
                # 업데이트할 필드와 값 설정
                update_fields = []
                params = []
                
                if drink_type:
                    update_fields.append("drink_type = ?")
                    params.append(drink_type)
                if temperature:
                    update_fields.append("temperature = ?")
                    params.append(temperature)
                if size:
                    update_fields.append("size = ?")
                    params.append(size)
                
                if not update_fields:
                    raise ValueError("수정할 필드가 없습니다.")
                
                # SQL 업데이트 쿼리 생성
                update_query = f"UPDATE orders SET {', '.join(update_fields)} WHERE order_id = ?"
                params.append(order_id)
                
                # 쿼리 실행
                cursor.execute(update_query, params)
                conn.commit()
                
                self._logger.info(f"주문 수정 완료: {order_id} - {', '.join(update_fields)}")
                
        except Exception as e:
            self._logger.error(f"주문 수정 중 오류 발생: {str(e)}")
            raise
    def add_option_to_drink(self, order_id: str, option: str, count: int = 1) -> None:
        """특정 음료에 옵션을 추가"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                
                # 주문 존재 확인
                cursor.execute('SELECT 1 FROM orders WHERE order_id = ?', (order_id,))
                if not cursor.fetchone():
                    raise ValueError(f"주문 ID {order_id}를 찾을 수 없습니다")
                
                # 옵션 추가
                for _ in range(count):
                    cursor.execute('''
                        INSERT INTO order_options (order_id, option_name)
                        VALUES (?, ?)
                    ''', (order_id, option))
                
                conn.commit()
                self._logger.info(f"옵션 추가 완료: 주문 {order_id}에 {option} {count}개")
                
        except Exception as e:
            self._logger.error(f"옵션 추가 중 오류 발생: {str(e)}")
            raise

    def remove_option_from_drink(self, order_id: str, option: str, count: int = 1) -> None:
        """특정 음료에서 옵션을 제거"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                
                # 옵션 ID 조회 (최신 순)
                cursor.execute('''
                    SELECT id FROM order_options 
                    WHERE order_id = ? AND option_name = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (order_id, option, count))
                
                option_ids = cursor.fetchall()
                
                # 옵션 제거
                for (option_id,) in option_ids:
                    cursor.execute('DELETE FROM order_options WHERE id = ?', (option_id,))
                
                conn.commit()
                self._logger.info(f"옵션 제거 완료: 주문 {order_id}에서 {option} {len(option_ids)}개")
                
        except Exception as e:
            self._logger.error(f"옵션 제거 중 오류 발생: {str(e)}")
            raise
    
    def get_drink_ids(self, conditions: Dict[str, Any]) -> List[str]:
        """조건에 맞는 주문 ID들을 찾아 반환
        
        Args:
            conditions: 검색 조건 딕셔너리 (예: {"drink_type": "아메리카노", "size": "미디움"})
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                
                # 동적 쿼리 생성
                query = "SELECT order_id FROM orders"
                params = []
                
                if conditions:
                    conditions_sql = []
                    for key, value in conditions.items():
                        if key == "additional_options":
                            # 추가 옵션은 별도 테이블 조인 필요
                            query += " JOIN order_options ON orders.order_id = order_options.order_id"
                            conditions_sql.append("order_options.option_name = ?")
                        else:
                            conditions_sql.append(f"{key} = ?")
                        params.append(value)
                    
                    if conditions_sql:
                        query += " WHERE " + " AND ".join(conditions_sql)
                
                # 중복 제거
                query = f"SELECT DISTINCT order_id FROM ({query})"
                
                cursor.execute(query, params)
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            self._logger.error(f"음료 ID 검색 중 오류 발생: {str(e)}")
            raise

    def get_drink(self, order_id: str) -> Optional[Drink]:
        """특정 주문 조회"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                
                # 주문 기본 정보 조회
                cursor.execute('''
                    SELECT drink_type, temperature, size
                    FROM orders WHERE order_id = ?
                ''', (order_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                drink = Drink()
                drink.order_id = order_id
                drink.drink_type, drink.temperature, drink.size = row
                
                # 옵션 조회
                cursor.execute('''
                    SELECT option_name FROM order_options
                    WHERE order_id = ?
                    ORDER BY created_at
                ''', (order_id,))
                
                drink.additional_options = [row[0] for row in cursor.fetchall()]
                return drink
                
        except Exception as e:
            self._logger.error(f"주문 조회 중 오류 발생: {str(e)}")
            raise
    
    def duplicate_order(self, order_id: str, count: int) -> List[str]:
        """주문을 지정된 수만큼 복제"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                
                # 원본 주문 정보 조회
                cursor.execute('''
                    SELECT drink_type, temperature, size
                    FROM orders WHERE order_id = ?
                ''', (order_id,))
                
                original = cursor.fetchone()
                if not original:
                    raise ValueError(f"복제할 주문을 찾을 수 없습니다: {order_id}")
                
                # 옵션 정보 조회
                cursor.execute('''
                    SELECT option_name
                    FROM order_options WHERE order_id = ?
                ''', (order_id,))
                options = [row[0] for row in cursor.fetchall()]
                
                # 주문 복제
                new_order_ids = []
                for _ in range(count):
                    new_order_id = self._generate_order_id()
                    new_order_ids.append(new_order_id)
                    
                    # 기본 주문 정보 복제
                    cursor.execute('''
                        INSERT INTO orders (order_id, drink_type, temperature, size)
                        VALUES (?, ?, ?, ?)
                    ''', (new_order_id, *original))
                    
                    # 옵션 복제
                    for option in options:
                        cursor.execute('''
                            INSERT INTO order_options (order_id, option_name)
                            VALUES (?, ?)
                        ''', (new_order_id, option))
                
                conn.commit()
                return new_order_ids
                
        except Exception as e:
            self._logger.error(f"주문 복제 중 오류 발생: {str(e)}")
            raise

    def get_order_summary(self) -> List[str]:
        """현재 주문 내역 요약"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                
                # 음료별 기본 정보와 옵션 조회 (옵션별 개수도 포함)
                cursor.execute('''
                    WITH order_options_count AS (
                        SELECT 
                            order_id,
                            option_name,
                            COUNT(*) as option_count
                        FROM order_options
                        GROUP BY order_id, option_name
                    ),
                    order_options_string AS (
                        SELECT 
                            o.order_id,
                            o.drink_type,
                            o.temperature,
                            o.size,
                            GROUP_CONCAT(
                                oc.option_name || oc.option_count || '번'
                                ORDER BY oc.option_name
                            ) as options_str
                        FROM orders o
                        LEFT JOIN order_options_count oc ON o.order_id = oc.order_id
                        GROUP BY o.order_id
                    )
                    SELECT 
                        drink_type,
                        temperature,
                        size,
                        options_str,
                        COUNT(*) as count
                    FROM order_options_string
                    GROUP BY drink_type, temperature, size, options_str
                ''')
                
                summary = []
                for row in cursor.fetchall():
                    drink_type, temp, size, options_str, count = row
                    if options_str:
                        summary.append(f"{temp} {drink_type} {size}사이즈 {options_str}추가 {count}잔")
                    else:
                        summary.append(f"{temp} {drink_type} {size}사이즈 {count}잔")
                
                return summary
                
        except Exception as e:
            self._logger.error(f"주문 요약 생성 중 오류 발생: {str(e)}")
            raise

    def delete_order(self, order_id: str) -> None:
        """특정 주문 ID의 주문을 삭제"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                # 주문 및 옵션 삭제
                cursor.execute('DELETE FROM order_options WHERE order_id = ?', (order_id,))
                cursor.execute('DELETE FROM orders WHERE order_id = ?', (order_id,))
                conn.commit()
                self._logger.info(f"주문 {order_id}이(가) 삭제되었습니다.")
        except Exception as e:
            self._logger.error(f"주문 삭제 중 오류 발생: {str(e)}")
            raise

        
    def clear_orders(self) -> None:
        """모든 주문 초기화"""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM order_options')
                cursor.execute('DELETE FROM orders')
                conn.commit()
                self._logger.info("모든 주문이 초기화되었습니다")
                
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

# 주문
class ActionOrderConfirmation(Action):
    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()
        
    def name(self) -> Text:
        return "action_order_confirmation"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # RASA 결과에서 엔티티 추출
            entities = self.mapper.extract_entities(tracker.latest_message)
            user_text = tracker.latest_message.get("text", "")
            
            # 로깅
            self._logger.info(f"주문 요청 - 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")
            
            # 음료 객체 생성 및 검증
            drink = self.mapper.create_drink_from_entities(entities)
            self.mapper.validate_drink(drink)
            
            # 주문 저장 (with 구문으로 트랜잭션 처리)
            order_ids = self.storage.add_order(drink)
            
            # 주문 요약
            order_summary = self.storage.get_order_summary()
            
            # 확인 메시지 생성
            confirmation_message = (
                f"주문이 접수되었습니다.\n"
                f"주문하신 음료: {drink}\n"
                f"현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            
            # 메시지 전송
            dispatcher.utter_message(text=confirmation_message)
            self._logger.info(f"주문 확인 완료 - IDs: {order_ids}")
            
            return []
            
        except ValueError as e:
            error_message = f"주문 처리 중 오류가 발생했습니다: {str(e)}"
            self._logger.error(error_message)
            dispatcher.utter_message(text=error_message)
            return []
            
        except Exception as e:
            error_message = "주문 처리 중 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
            return []

# 주문 테스트             

# 주문 테스트 케이스
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

        
# 주문 변경
class ActionModifyOrder(Action):
    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()
        
    def name(self) -> Text:
        return "action_modify_order"
    
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # RASA 결과에서 엔티티 추출
            entities = tracker.latest_message.get("entities", [])
            user_text = tracker.latest_message.get("text", "")
            self._logger.info(f"주문 변경 요청 - 사용자 입력: {user_text}")

             # 변경 대상과 새로운 음료 분리
            split_keywords = ["대신", "말고", "은", "는", "을", "를"]
            split_pattern = '|'.join(split_keywords)
            split_text = re.split(f"({split_pattern})", user_text)
            
            if len(split_text) < 2:
                raise ValueError("변경할 음료와 새로운 음료를 구분할 수 없습니다.")
                    
            # 변경 대상 음료명 추출
            target_drink_text = split_text[0].strip()
            new_drink_text = split_text[-1].strip()

            # 타겟 음료 엔티티 추출
            target_entities = []
            for entity in entities:
                entity_text = user_text[entity['start']:entity['end']]
                if entity_text in target_drink_text:
                    target_entities.append(entity)

            # 새로운 음료 엔티티 추출
            new_entities = []
            for entity in entities:
                entity_text = user_text[entity['start']:entity['end']]
                if entity_text in new_drink_text:
                    new_entities.append(entity)

            # 음료 객체 생성
            target_drink = self.mapper.create_drink_from_entities(target_entities)
            new_drink = self.mapper.create_drink_from_entities(new_entities)


            # 기존 주문에서 속성 복사 (size, temperature, additional_options 등)
            if not new_drink.size and target_drink.size:
                new_drink.size = target_drink.size
            if not new_drink.temperature and target_drink.temperature:
                new_drink.temperature = target_drink.temperature
            if not new_drink.additional_options and target_drink.additional_options:
                new_drink.additional_options = target_drink.additional_options

            self.mapper.validate_drink(new_drink)

            # 검색 조건 생성 - 모든 관련 속성 포함
            search_conditions = {}
            for entity in target_entities:
                if entity['entity'] == 'drink_type':
                    search_conditions["drink_type"] = entity['value']
                elif entity['entity'] == 'temperature':
                    search_conditions["temperature"] = entity['value']
                elif entity['entity'] == 'size':
                    search_conditions["size"] = entity['value']
                elif entity['entity'] == 'option':
                    search_conditions["option"] = entity['value']

            if not search_conditions:
                raise ValueError("변경할 음료의 조건을 찾을 수 없습니다.")

            # 조건에 맞는 주문 ID들 조회
            order_ids = self.storage.get_drink_ids(search_conditions)
            if not order_ids:
                conditions_str = ", ".join(f"{k}: {v}" for k, v in search_conditions.items())
                raise ValueError(f"변경할 주문을 찾을 수 없습니다. 검색 조건: {conditions_str}")

            # 새로운 음료 객체 생성 - 기존 속성 유지
            new_drink = self.mapper.create_drink_from_entities(new_entities)
            # 변경되지 않은 속성은 기존 주문의 첫 번째 음료에서 복사
            first_order = self.storage.get_drink(order_ids[0])
            if first_order:
                if not new_drink.size:
                    new_drink.size = first_order.size
                if not new_drink.temperature:
                    new_drink.temperature = first_order.temperature
                if not new_drink.additional_options:
                    new_drink.additional_options = first_order.additional_options
            
            # 수량 처리 - 기존 주문과 새로운 주문의 수량 비교
            quantity = next((int(entity['value']) for entity in new_entities 
                           if entity['entity'] == 'quantity'), len(order_ids))
            
            # 변경하려는 수량이 기존 주문 수량보다 많은 경우 추가 주문 생성
            if quantity > len(order_ids):
                additional_needed = quantity - len(order_ids)
                new_ids = self.storage.duplicate_order(order_ids[0], additional_needed)
                order_ids.extend(new_ids)

            #  주문 변경 실행
            modified_count = 0
            for order_id in order_ids[:quantity]:
                self.storage.modify_drink(
                    order_id,
                    drink_type=new_drink.drink_type if new_drink.drink_type else None,
                    temperature=new_drink.temperature if new_drink.temperature else None,
                    size=new_drink.size if new_drink.size else None
                )
                
                # 옵션 처리
                if new_drink.additional_options:
                    # 기존 옵션 제거
                    current_options = self.storage.get_drink_options(order_id)
                    for option in current_options:
                        self.storage.remove_option_from_drink(order_id, option)
                    
                    # 새로운 옵션 추가
                    for option in new_drink.additional_options:
                        self.storage.add_option_to_drink(order_id, option)
                
                modified_count += 1


            # 메시지 전송
            # 주문 변경 확인 메시지 생성
            # 주문 요약 조회
            order_summary = self.storage.get_order_summary()
            
            confirmation_message = (
                f"주문이 변경되었습니다.\n"  # 주문 접수 -> 변경 메시지로 수정
                f"변경된 음료: {modified_count}잔\n"  # 변경된 음료 수량 표시
                f"현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"  # 최신 주문 내역 조회
            )
            
            dispatcher.utter_message(text=confirmation_message)
            self._logger.info(f"주문 변경 완료 - 변경된 음료 수: {modified_count}")

            return []
            
        except ValueError as e:
            error_message = f"주문 변경 중 오류가 발생했습니다: {str(e)}"
            self._logger.error(error_message)
            dispatcher.utter_message(text=error_message)
            return []
            
        except Exception as e:
            error_message = "주문 변경 중 오류가 발생했습니다. 다시 시도해주세요."
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

# 주문 제거
class ActionSubtractFromOrder(Action):
    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()

    def name(self) -> Text:
        return "action_subtract_from_order"
    
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        try:
            # RASA 결과 및 엔티티 추출
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            user_text = tracker.latest_message.get("text", "")
            logging.info(f"(주문 제거) RASA 분류 결과: {entities}")

            # 표준화 작업
            drink = self.mapper.create_drink_from_entities(entities)
            logging.info(f"(주문 제거) 표준화된 엔티티: {drink}")

            # 검색 조건 생성
            conditions = {}
            if drink.drink_type:
                conditions["drink_type"] = drink.drink_type
            if drink.temperature:
                conditions["temperature"] = drink.temperature
            if drink.size:
                conditions["size"] = drink.size

            if not conditions:
                raise ValueError("제거할 음료의 조건을 찾을 수 없습니다.")

            # 조건에 맞는 주문 ID 검색
            order_ids = self.storage.get_drink_ids(conditions)
            
            # 제거할 수량 결정
            quantity_to_remove = drink.quantity if drink.quantity else 1
            current_quantity = len(order_ids)

            if quantity_to_remove > current_quantity:
                raise ValueError(f"현재 주문 수량({current_quantity}잔)보다 많은 수량({quantity_to_remove}잔)을 삭제할 수 없습니다.")
            


            # 주문 제거
            removed_count = 0
            for order_id in order_ids[:quantity_to_remove]:
                self.storage.delete_order(order_id)
                removed_count += 1
                self._logger.info(f"{order_id} 주문이 제거되었습니다. 현재 제거된 잔 수: {removed_count}")

            # 주문 요약 정보 출력
            order_summary = self.storage.get_order_summary()
            
            confirmation_message = (
                f"{drink.drink_type} {removed_count}잔이 주문에서 제거되었습니다.\n"
                f"현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            
            dispatcher.utter_message(text=confirmation_message)
            self._logger.info(f"주문 제거 완료 - 제거된 음료 수: {removed_count}")

            return []

            
            # 현재 주문 요약 정보 출력
            order_summary = ', '.join(self.storage.get_order_summary())
            confirmation_message = f"현재 주문은 {order_summary}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)

            return []

        except ValueError as e:
            logging.error(f"주문 처리 중 오류 발생: {e}")
            dispatcher.utter_message(text="주문 처리 중 오류가 발생했습니다. 다시 시도해주세요.")
            return []  
        except Exception as e:
            logging.error(f"주문 처리 중 오류 발생: {e}")
            dispatcher.utter_message(text="주문 처리 중 오류가 발생했습니다. 다시 시도해주세요.")
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
    def __init__(self):
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()
        self._logger = LoggerSetup.get_logger()

    def name(self) -> Text:
        return "action_add_subtract"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # RASA 결과에서 엔티티 추출
            entities = tracker.latest_message.get("entities", [])
            user_text = tracker.latest_message.get("text", "")
            self._logger.info(f"주문 수량 변경 요청 - 사용자 입력: {user_text}")
            
            # 음료 객체 생성을 통한 표준화
            drink = self.mapper.create_drink_from_entities(entities)
            self._logger.info(f"표준화된 음료 정보: {drink}")
            
            # 검색 조건 생성 (사용자가 명시한 속성만)
            search_conditions = {}
            for entity in entities:
                if drink.drink_type:
                    search_conditions["drink_type"] = drink.drink_type
                elif drink.temperature:
                    search_conditions["temperature"] = drink.temperature
                elif drink.size:
                    search_conditions["size"] = drink.size

            if not search_conditions:
                raise ValueError("변경할 음료의 조건을 찾을 수 없습니다.")

            # 조건에 맞는 주문 ID들 조회
            order_ids = self.storage.get_drink_ids(search_conditions)
            if not order_ids:
                conditions_str = ", ".join(f"{k}: {v}" for k, v in search_conditions.items())
                raise ValueError(f"변경할 주문을 찾을 수 없습니다. 검색 조건: {conditions_str}")

            # 기존 주문의 첫 번째 음료 정보 가져오기
            first_order = self.storage.get_drink(order_ids[0])
            
            # 수량 변경 처리
            quantity_change = drink.quantity if drink.quantity else 1
            current_quantity = len(order_ids)

            if "추가" in user_text or "더" in user_text:
                # 주문 추가
                new_ids = self.storage.duplicate_order(order_ids[0], quantity_change)
                modified_count = len(new_ids)
                action_type = "추가"
            else:
                # 주문 감소
                if quantity_change > current_quantity:
                    raise ValueError(f"현재 주문 수량({current_quantity}잔)보다 많은 수량({quantity_change}잔)을 삭제할 수 없습니다.")
                
                for order_id in order_ids[:quantity_change]:
                    self.storage.delete_order(order_id)
                modified_count = quantity_change
                action_type = "삭제"

            # 주문 요약 조회
            order_summary = self.storage.get_order_summary()

            # 메시지 생성
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
    def __init__(self):
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()
        self._logger = LoggerSetup.get_logger()
    
    def name(self) -> Text:
        return "action_order_finish"
    
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # rasa 결과 추출
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            user_text = tracker.latest_message.get("text", "")
            drink = self.mapper.create_drink_from_entities(entities)

            # 현재 주문 내역 요약 조회
            order_summary = self.storage.get_order_summary()
            
            # 주문 요약 메시지 생성
            confirmation_message = (
                f"현재 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}\n"
                f"주문을 완료하시겠습니까?"
            )
            
            # 메시지 전송
            dispatcher.utter_message(text=confirmation_message)

        except ValueError as e:
            logging.error(f"주문 처리 중 오류 발생: {e}")
            dispatcher.utter_message(text="주문 처리 중 오류가 발생했습니다. 다시 시도해주세요.")
            return []  
        except Exception as e:
            logging.error(f"주문 처리 중 오류 발생: {e}")
            dispatcher.utter_message(text="주문 처리 중 오류가 발생했습니다. 다시 시도해주세요.")  
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
    def __init__(self):
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()
        self._logger = LoggerSetup.get_logger()

    def name(self) -> Text:
        return "action_cancel_order"
    
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        try:
            # RASA 결과 및 엔티티 추출
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            user_text = tracker.latest_message.get("text", "")
            logging.info(f"(주문 취소) RASA 분류 결과: {entities}")
            # 모든 주문 초기화
            self.storage.clear_orders()
            # 사용자에게 주문 초기화 확인 메시지 전달
            dispatcher.utter_message(text="주문이 초기화되었습니다.")

            return []

        except ValueError as e:
            logging.error(f"주문 처리 중 오류 발생: {e}")
            dispatcher.utter_message(text="주문 처리 중 오류가 발생했습니다. 다시 시도해주세요.")
            return []  
        except Exception as e:
            logging.error(f"주문 처리 중 오류 발생: {e}")
            dispatcher.utter_message(text="주문 처리 중 오류가 발생했습니다. 다시 시도해주세요.")
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
    def __init__(self):
        self._logger = LoggerSetup.get_logger()
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()

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

            # 표준화 작업
            drink = self.mapper.create_drink_from_entities(entities)
            self._logger.info(f"(사이즈 변경) 표준화된 음료: {drink}")

            if not drink.size:
                raise ValueError("변경할 사이즈 정보가 입력되지 않았습니다.")
            if not drink.drink_type:
                raise ValueError("변경할 음료 종류가 입력되지 않았습니다.")

            # 2. 사이즈 및 음료 종류 엔티티 추출 및 표준화
            size_entity = next((e for e in entities if e["entity"] == "size"), None)
            drink_type_entity = next((e for e in entities if e["entity"] == "drink_type"), None)

            if not size_entity:
                raise ValueError("사이즈 정보가 입력되지 않았습니다.")
            if not drink_type_entity:
                raise ValueError("변경할 음료 종류가 입력되지 않았습니다.")

            # 검색 조건 생성 (사용자가 명시한 속성만)
            search_conditions = {}
            for entity in entities:
                if drink.drink_type:
                    search_conditions["drink_type"] = drink.drink_type
                elif drink.temperature:
                    search_conditions["temperature"] = drink.temperature
                elif drink.size:
                    search_conditions["size"] = drink.size

            # 조건에 맞는 주문 ID 검색
            order_ids = self.storage.get_drink_ids(search_conditions)
            if not order_ids:
                conditions_str = ", ".join(f"{k}: {v}" for k, v in search_conditions.items())
                raise ValueError(f"해당하는 주문({conditions_str})을 찾을 수 없습니다.")

            # 모든 해당하는 음료의 사이즈 변경
            modified_drinks = []
            for order_id in order_ids:
                current_drink = self.storage.get_drink(order_id)
                current_drink.size = drink.size
                
                self.storage.modify_drink(
                    order_id, 
                    drink_type=current_drink.drink_type,
                    temperature=current_drink.temperature,
                    size=current_drink.size
                )
                modified_drinks.append(current_drink)
                self._logger.info(f"사이즈가 '{drink.size}'로 변경되었습니다 - 주문 ID: {order_id}")

            # 주문 요약 정보 출력
            order_summary = self.storage.get_order_summary()
            modified_drinks_str = "\n".join([f"- {drink}" for drink in modified_drinks])
            confirmation_message = (
                f"총 {len(modified_drinks)}개의 '{drink.drink_type}' 음료의 사이즈가 '{drink.size}'로 변경되었습니다.\n"
                f"\n변경된 음료 목록:\n{modified_drinks_str}\n"
                f"\n현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            dispatcher.utter_message(text=confirmation_message)

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
            # 1. 사용자 입력에서 엔티티 추출
            entities = [entity for entity in tracker.latest_message.get("entities", []) 
                       if entity.get("extractor") != "DIETClassifier"]
            user_text = tracker.latest_message.get("text", "")
            self._logger.info(f"(온도 변경 요청) 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")

            # 표준화 작업
            drink = self.mapper.create_drink_from_entities(entities)
            self._logger.info(f"(온도 변경) 표준화된 음료: {drink}")

            if not drink.drink_type:
                raise ValueError("변경할 음료 종류가 입력되지 않았습니다.")

            # 온도가 입력되지 않은 경우 현재 온도의 반대로 변경
            if not drink.temperature:
                # 현재 주문된 음료 찾기
                drink_ids = self.storage.get_drink_ids({"drink_type": drink.drink_type})
                if not drink_ids:
                    raise ValueError(f"'{drink.drink_type}'에 대한 주문이 없습니다.")
                current_drink = self.storage.get_drink(drink_ids[-1])
                
                # 현재 온도의 반대로 설정
                if current_drink.temperature == "아이스":
                    drink.temperature = "핫"
                else:
                    drink.temperature = "아이스"
                self._logger.info(f"온도가 {drink.temperature}으로 변경되었습니다.")

            # 검색 조건 생성 (사용자가 명시한 속성만)
            conditions = {}
            for entity in entities:
                if drink.drink_type:
                    conditions["drink_type"] = drink.drink_type
                elif drink.temperature:
                    conditions["temperature"] = drink.temperature
                elif drink.size:
                    conditions["size"] = drink.size

            # 조건에 맞는 주문 ID 검색
            order_ids = self.storage.get_drink_ids(conditions)
            if not order_ids:
                conditions_str = ", ".join(f"{k}: {v}" for k, v in conditions.items())
                raise ValueError(f"해당하는 주문({conditions_str})을 찾을 수 없습니다.")

            # 모든 해당하는 음료의 온도 변경
            modified_drinks = []
            for order_id in order_ids:
                current_drink = self.storage.get_drink(order_id)
                current_drink.temperature = drink.temperature
                
                self.storage.modify_drink(
                    order_id, 
                    drink_type=current_drink.drink_type,
                    temperature=current_drink.temperature,
                    size=current_drink.size
                )
                modified_drinks.append(current_drink)
                self._logger.info(f"온도가 '{drink.temperature}'로 변경되었습니다 - 주문 ID: {order_id}")

            # 주문 요약 정보 출력
            order_summary = self.storage.get_order_summary()
            modified_drinks_str = "\n".join([f"- {drink}" for drink in modified_drinks])
            confirmation_message = (
                f"총 {len(modified_drinks)}개의 '{drink.drink_type}' 음료의 온도가 '{drink.temperature}'로 변경되었습니다.\n"
                f"\n변경된 음료 목록:\n{modified_drinks_str}\n"
                f"\n현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            dispatcher.utter_message(text=confirmation_message)

            return []

        except ValueError as e:
            error_message = f"온도 변경 중 오류 발생: {str(e)}"
            self._logger.error(error_message)
            dispatcher.utter_message(text=error_message)
            return []
        except Exception as e:
            error_message = "온도 변경 중 예상치 못한 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
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
    def name(self) -> Text:
        return "action_add_additional_option"
    
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # rasa 결과 추출
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            user_text = tracker.latest_message.get("text", "")

            # 로그 출력
            logging.info(f"(커피 추가옵션 추가) rasa 결과: {entities}")

            # 유저에게 확인 메시지 전달
            confirmation_message = f"의도: 커피 추가옵션 추가 \n 엔티티: {entities} \n 유저 텍스트: {user_text}"
            dispatcher.utter_message(text=confirmation_message)

        except ValueError as e:
            logging.error(f"주문 처리 중 오류 발생: {e}")
            dispatcher.utter_message(text="주문 처리 중 오류가 발생했습니다. 다시 시도해주세요.")
            return []  
        except Exception as e:
            logging.error(f"주문 처리 중 오류 발생: {e}")
            dispatcher.utter_message(text="주문 처리 중 오류가 발생했습니다. 다시 시도해주세요.")  
            return []

# 커피 추가옵션 제거        
class ActionRemoveAdditionalOption(Action):
    def __init__(self):
        self.storage = OrderStorage()
        self.mapper = DrinkMapper()
        self._logger = LoggerSetup.get_logger()

    def name(self) -> Text:
        return "action_remove_additional_option"
    
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # rasa 결과 추출
            entities = [entity for entity in tracker.latest_message.get("entities", []) 
                       if entity.get("extractor") != "DIETClassifier"]
            user_text = tracker.latest_message.get("text", "")
            self._logger.info(f"(옵션 제거 요청) 사용자 입력: {user_text}")
            self._logger.debug(f"추출된 엔티티: {entities}")
            self._logger.info(f"주문 수량 변경 요청 - 사용자 입력: {user_text}")
            
            # 음료 객체 생성을 통한 표준화
            drink = self.mapper.create_drink_from_entities(entities)
            self._logger.info(f"(옵션 제거) 표준화된 음료: {drink}")

            if not drink.additional_options:
                raise ValueError("제거할 옵션이 입력되지 않았습니다.")
            if not drink.drink_type:
                raise ValueError("변경할 음료 종류가 입력되지 않았습니다.")
            
            # 검색 조건 생성 (사용자가 명시한 속성만)
            search_conditions = {}
            for entity in entities:
                if drink.drink_type:
                    search_conditions["drink_type"] = drink.drink_type
                elif drink.temperature:
                    search_conditions["temperature"] = drink.temperature 
                elif drink.size:
                    search_conditions["size"] = drink.size
                elif drink.additional_options:  # 추가 옵션이 있는 경우
                    search_conditions["additional_options"] = drink.additional_options

            # 조건에 맞는 주문 ID 검색
            order_ids = self.storage.get_drink_ids(search_conditions)
            if not order_ids:
                conditions_str = ", ".join(f"{k}: {v}" for k, v in search_conditions.items())
                raise ValueError(f"해당하는 주문({conditions_str})을 찾을 수 없습니다.")
            
            # 가장 최근 주문 선택
            order_id = order_ids[-1]
            current_drink = self.storage.get_drink(order_id)
            
            # 제거할 옵션 수량 결정
            options_to_remove = {}
            for option in drink.additional_options:
                option_name = option
                current_count = sum(1 for opt in current_drink.additional_options 
                                  if opt == option_name)
                
                if current_count == 0:
                    continue
                    
                # quantity 엔티티가 있는지 확인
                quantity_entity = {"value": drink.quantity} if drink.quantity else None  # drink 객체에서 수량 정보 가져오기
                if quantity_entity:
                    # 수량이 지정된 경우
                    remove_count = quantity_entity["value"]  
                    options_to_remove[option_name] = min(remove_count, current_count)
                else:
                    # 수량이 지정되지 않은 경우 모두 제거
                    options_to_remove[option_name] = current_count

            if not options_to_remove:
                raise ValueError(f"'{current_drink.drink_type}'에서 제거할 수 있는 옵션이 없습니다.")

            # 옵션 제거 실행
            removed_options = []
            for option_name, count in options_to_remove.items():
                for _ in range(count):
                    self.storage.remove_option_from_drink(order_id, option_name)
                    removed_options.append(f"{option_name} 1개")
                self._logger.info(f"옵션 '{option_name}' {count}개가 제거되었습니다 - 주문 ID: {order_id}")

            # 변경된 음료 정보 가져오기
            updated_drink = self.storage.get_drink(order_id)

            # 주문 요약 정보 출력
            order_summary = self.storage.get_order_summary()
            removed_options_str = ", ".join(removed_options)
            confirmation_message = (
                f"음료 '{current_drink.drink_type}'에서 {removed_options_str}가 제거되었습니다.\n"
                f"변경된 음료: {updated_drink}\n"
                f"\n현재 전체 주문 내역:\n"
                f"{chr(10).join(f'- {item}' for item in order_summary)}"
            )
            dispatcher.utter_message(text=confirmation_message)

            return []

        except ValueError as e:
            error_message = f"옵션 제거 중 오류 발생: {str(e)}"
            self._logger.error(error_message)
            dispatcher.utter_message(text=error_message)
            return []
        except Exception as e:
            error_message = "옵션 제거 중 예상치 못한 오류가 발생했습니다. 다시 시도해주세요."
            self._logger.error(f"예상치 못한 오류: {str(e)}")
            dispatcher.utter_message(text=error_message)
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
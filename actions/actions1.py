"""
This files contains your custom actions which can be used to run
custom Python code.

See this guide on how to implement these action:
https://rasa.com/docs/rasa/custom-actions


This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionChatGPT(Action):

    def name(self) -> Text:
        return "action_chatGPT"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []
"""
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import logging
import re
import copy

# 로거 설정
logging.basicConfig(level=logging.DEBUG)

# 한국어 수량을 숫자로 변환하는 메서드
# dictionary.get(key, default), 딕셔너리에서 :의 왼쪽은 키값(key) 오른쪽은 값(value)이다
def korean_to_number(korean: str) -> int:
    korean_number_map = {
        "한": 1,
        "두": 2,
        "세": 3,
        "네": 4,
        "다섯": 5,
        "여섯": 6,
        "일곱": 7,
        "여덟": 8,
        "아홉": 9,
        "열": 10
    }
    return korean_number_map.get(korean, 1)  # korean 문자열이 사전에 존재하지 않으면 기본값으로 1을 반환

# 숫자를 한국어 수량으로 변환하는 메서드
def number_to_korean(number: int) -> str:
    number_korean_map = {
        1: "한",
        2: "두",
        3: "세",
        4: "네",
        5: "다섯",
        6: "여섯",
        7: "일곱",
        8: "여덟",
        9: "아홉",
        10: "열"
    }
    return number_korean_map.get(number, str(number))  # 기본값으로 숫자 문자열을 반환

# 음료 종류 및 띄어쓰기 표준화 메서드
def standardize_drink_name(name):
    # 음료 이름 변형을 표준 이름으로 매핑하는 사전
    drink_name_map = {
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
        "라때" : "카페라떼"
    }

    # 공백과 쉼표 제거
    # re.sub는 (패턴, 교체할 값, 검색할 대상 문자열, 교체할 최대횟수(선택), 동작 수정 플래그(선택)) 으로 동작한다.
    standardized_name = re.sub(r'[\s,]+', '', name)

    # 매핑 사전을 사용하여 표준 이름으로 변환
    if standardized_name in drink_name_map:
        return drink_name_map[standardized_name]
    else:
        return standardized_name

# 온도를 표준화하는 메서드
def standardize_temperature(value):
    if value in ["차갑게", "시원하게", "아이스", "차가운", "시원한"]:
        return "아이스"
    elif value in ["뜨겁게", "따뜻하게", "핫", "뜨거운", "따뜻한", "뜨뜻한", "하수", "hot"]:
        return "핫"
    return value

# 잔 수를 표준화하는 메서드
def standardize_quantity(value):
    numeral_map = {
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
    value = value.strip()
    if value.isdigit():
        return int(value)
    else:
        return numeral_map.get(value, 1)

# 사이즈를 표준화하는 메서드
def standardize_size(value):
    if value in ["미디움", "보통", "중간", "기본", "톨", "비디오", "토"]:
        return "미디움"
    elif value in ["라지", "큰", "크게", "라의", "라디오", "라디"]:
        return "라지"
    elif value in ["엑스라지", "엑스라이즈", "제1 큰", "가장 큰", "제1 크게", "맥시멈"]:
        return "엑스라지"
    return value

# 추가옵션를 표준화하는 메서드
def standardize_option(value):
    if value in ["샤츠", "셔츠", "사추", "샤타나", "4추가"]:
        return "샷"
    elif value in ["카라멜실업", "실룩실룩", "가라멜시럽", "카라멜시로"]:
        return "카라멜시럽"
    elif value in ["바닐라실업"]:
        return "바닐라시럽"
    elif value in ["비비크림"]:
        return "휘핑크림"
    return value

# 테이크아웃을 표준화하는 메서드
def standardize_take(value):
    if value in ["테이크아웃", "들고", "가져", "먹을", "마실", "아니요"]:
        return "포장"
    elif value in ["먹고", "여기", "이곳", "네"]:
        return "매장"
    return value

# 커피의 종류가 정해지지 않으면 오류 발생 메서드
def raise_missing_attribute_error(drinks):
    if not drinks:
        raise ValueError("정확한 음료의 종류를 말씀하여주세요.")
import uuid

class Drink:
    _id_counter = 1

    def __init__(self, drink_type, order_id=None, temperature=None, size=None, additional_options=None, quantity=1):
        self.order_id = Drink._id_counter
        Drink._id_counter += 1
        self.drink_type = drink_type
        self.temperature = temperature
        self.size = size
        self.quantity = quantity
        self.additional_options = additional_options or []

# 현재 주문 목록을 저장
class OrderManager:
    def __init__(self):
        self.orders = []
        self.hot_drinks = ["허브티"]  # 항상 핫으로만 제공되는 음료 리스트
        self.ice_only_drinks = ["토마토주스", "키위주스", "망고스무디", "딸기스무디", "레몬에이드", "복숭아아이스티"]  # 항상 아이스로만 제공되는 음료 리스트
        self.temperatures = {}
        self.sizes = {}
        self.additional_option = {}
        self.order_id_counter = 0

    # 주문을 추가하는 메서드
    def add_order(self, drink_type: Text, quantity: int, temperature: Text, size: Text, additional_options: List[Text]=None):
        drink_type = standardize_drink_name(drink_type)  # 음료 이름을 표준화

        # 아이스로만 제공되는 음료일 경우
        if drink_type in self.ice_only_drinks:
            temperature = "아이스"
            logging.info(f"{drink_type}는(은) 온도를 변경할 수 없습니다. 아이스로 제공됩니다.")
        # 핫으로만 제공되는 음료일 경우
        elif drink_type in self.hot_drinks:
            temperature = "핫"
            logging.info(f"{drink_type}는(은) 온도를 변경할 수 없습니다. 핫으로 제공됩니다.")

        drink = Drink(
            drink_type=drink_type,
            quantity=quantity,
            temperature=temperature,
            size=size,
            additional_options=additional_options or []
        )
        self.orders.append(drink)
        logging.info(f"{drink_type} {quantity}잔이 {temperature}로 추가되었습니다.")

    # 주문 검색 메서드
    def find_orders(self, drink_type=None, temperature=None, size=None, additional_options=None):
        results = []
        for order in self.orders:
            if (drink_type is None or order.drink_type == drink_type) and \
               (temperature is None or order.temperature == temperature) and \
               (size is None or order.size == size) and \
               (additional_options is None or set(additional_options).issubset(set(order.additional_options))):
                results.append(order)
        return results
    
    def find_order(self, drink_type=None, temperature=None, size=None, additional_options=None):
        for order in self.orders:
            if (drink_type is None or order.drink_type == drink_type) and \
               (temperature is None or order.temperature == temperature) and \
               (size is None or order.size == size) and \
               (additional_options is None or set(additional_options).issubset(set(order.additional_options))):
                return order
        return None
    
    # 커피 추가 메서드
    

    # 커피 변경 메서드
    def modify_order(self, order_id, new_drink_type=None, temperature=None, size=None, additional_options=None):
        new_drink_type = standardize_drink_name(new_drink_type)  # 음료 이름 표준화
        order = next((o for o in self.orders if o.order_id == order_id), None)
        # 기존 주문을 새로운 주문으로 수정하는 메서드
        if order:
            if new_drink_type:
                order.drink_type = new_drink_type
            if temperature:
                order.temperature = temperature
            if size:
                order.size = size
            if additional_options:
                order.additional_options = additional_options
        else:
            raise ValueError(f"주문 ID {order_id}를 찾을 수 없습니다.")
    def get_orders(self):
        return self.orders


    # 커피 제거 메서드
    def subtract_order(self, order_id):
        self.orders = [order for order in self.orders if order.order_id != order_id]
        
    def update_order(self, updated_order):
        for idx, order in enumerate(self.orders):
            if order.order_id == updated_order.order_id:
                self.orders[idx] = updated_order
                break
        
    
    # 커피 추가옵션 추가 메서드
    def add_additional_options(self, order_id, new_options):
        # 음료 이름을 표준화하여 데이터베이스의 일관성과 맞추기
        new_options = standardize_drink_name(new_options)
        logging.warning(f"추가옵션 추가 실행")  # 추가 옵션 추가 작업 시작을 로그에 기록
        logging.warning(f"현재 상태: 주문 수 - {len(self.orders)}, 총 주문 - {[order.drink_type for order in self.orders]}")
        order = next((o for o in self.orders if o.order_id == order_id), None)
        if order:
            logging.warning(f"주문 ID {order_id}에 새로운 옵션 추가 중: {new_options}")
            order.additional_options.extend(new_options)
            order.additional_options = list(set(order.additional_options))  # 중복 제거
            logging.warning(f"주문 ID {order_id}의 최종 추가 옵션: {order.additional_options}")
        else:
            raise ValueError(f"주문 ID {order_id}를 찾을 수 없습니다.")

    # 커피 추가옵션 제거 메서드
    def remove_additional_options(self, order_id, options_to_remove):
        logging.warning(f"추가옵션 제거 실행")
        order = next((o for o in self.orders if o.order_id == order_id), None)
        if order:
            logging.warning(f"제거 전 추가 옵션: {order.additional_options}")
            order.additional_options = [opt for opt in order.additional_options if opt not in options_to_remove]
            logging.warning(f"제거 후 추가 옵션: {order.additional_options}")
        else:
            raise ValueError(f"주문 ID {order_id}를 찾을 수 없습니다.")
    # 주문 취소 메서드
    def cancel_order(self):
        # 현재 모든 주문을 취소하고 초기화하는 메서드
        canceled_orders = self.orders.copy()  # 기존 주문을 백업
        self.orders = []
        self.temperatures = {}
        self.sizes = {}
        self.additional_option = {}
        return canceled_orders  # 취소된 주문 반환

    # 주문 내역 초기화 메서드
    def clear_order(self):
        # 현재 주문을 초기화하는 메서드
        self.orders.clear()
        self.temperatures.clear()
        self.sizes.clear()
        self.additional_option.clear()

    # 주문 내역 반환 메서드
    def get_order_list(self):
        order_list = []
        for idx, order in enumerate(self.orders, 1):
            options = ", ".join(order.additional_options) if order.additional_options else "없음"
            order_info = f"{idx}. [ID: {order.order_id}] {order.temperature} {order.size} {order.drink_type} (추가 옵션: {options})"
            order_list.append(order_info)
        return order_list

    # 주문 확인 후 출력 메서드
    def get_order_summary(self):
        summary = []
        for order in self.orders:
            options = ', '.join(order.additional_options) if order.additional_options else '옵션 없음'
            summary.append(f"{order.quantity}잔의 {order.temperature} {order.drink_type} ({order.size}, {options})")
        return summary
    
    def generate_order_id(self):
        self.order_id_counter += 1
        return f"order_{self.order_id_counter}"


order_manager = OrderManager()  # OrderManager 인스턴스 생성

# 엔티티 매핑
class OrderMapper:
    def __init__(self, entities, is_temperature_change=False, is_size_change=False):
        self.entities = sorted(entities, key=lambda x: x['start'])
        self.is_temperature_change = is_temperature_change  # 온도 변경 기능 실행 여부 플래그
        self.is_size_change = is_size_change  # 사이즈 변경 기능 실행 여부 플래그
        self.drinks: List[Drink] = []  # 매핑된 음료 목록
        self.suffixes = ("사이즈로", "사이즈", "으로", "으", "걸로", "로", "는", "은", "해주세요", "해서", "해", "한거", "이랑", "도")
        self._map_entities()

    # 주문 초기화 메서드
    def _initialize_order(self) -> Drink:
        """
        새로운 음료 주문을 초기화하는 메서드
        """
        return Drink(
            drink_type=None,
            order_id=str(uuid.uuid4()),
            temperature=None,
            size=None,
            quantity=1,
            additional_options=[]
        )

    
    def find_order(self, drink_type=None, temperature=None, size=None, additional_options=None):
        for order in self.orders:
            if  (drink_type is None or order.drink_type == drink_type) and \
                (temperature is None or order.temperature == temperature) and \
                (size is None or order.size == size) and \
                (additional_options is None or set(additional_options).issubset(set(order.additional_options))):
                return order
        return None

    # 엔티티 값에서 제거할 접미사 제거 메서드
    def clean_entity_values(self):
        """
        엔티티 값에서 불필요한 접미사를 제거하는 메서드
        """
        for entity in self.entities:
            if entity['entity'] != 'drink_type':
                value = entity["value"]
                for suffix in self.suffixes:
                    if value.endswith(suffix):
                        value = value[:-len(suffix)]
                        break
                entity["value"] = value

    # 음료와 온도, 잔 수, 사이즈, 추가옵션 매핑 메서드
    def _map_entities(self):
        """
        엔티티를 순차적으로 처리하여 각 음료의 속성을 매핑하는 메서드
        """
        self.clean_entity_values()  # 엔티티 값 정리
        current_drink = self._initialize_order()  # 현재 처리 중인 주문 초기화

        # 온도 엔티티를 큐로 관리하여 음료에 순차적으로 할당
        temperature_queue = []

        for entity in self.entities:
            entity_type = entity['entity']
            entity_value = entity['value']

            if entity_type == 'temperature':
                # 온도 엔티티를 큐에 추가
                temperature = self._map_temperature(entity_value)
                temperature_queue.append(temperature)
                logging.debug(f"온도 엔티티 큐에 추가: {temperature}")

            elif entity_type == 'drink_type':
                if current_drink.drink_type:
                    self._complete_order(current_drink)  # 현재 음료 주문 완료 및 추가
                    current_drink = self._initialize_order()  # 새로운 음료 주문 초기화

                # 음료 타입 설정
                current_drink.drink_type = standardize_drink_name(entity_value)
                logging.debug(f"음료 타입 설정: {current_drink.drink_type}")

                # 큐에서 온도를 가져와 음료에 할당
                if temperature_queue:
                    current_drink.temperature = temperature_queue.pop(0)
                    logging.debug(f"온도 할당: {current_drink.temperature}")
                else:
                    # 온도 엔티티가 없을 경우 기본 온도 설정
                    current_drink.temperature = self._set_default_temperature(current_drink.drink_type)
                    logging.debug(f"기본 온도 할당: {current_drink.temperature}")

            elif entity_type == 'quantity':
                current_drink.quantity = standardize_quantity(entity_value)
                logging.debug(f"잔 수 설정: {current_drink.quantity}")

            elif entity_type == 'size':
                current_drink.size = self._map_size(entity_value)
                logging.debug(f"사이즈 설정: {current_drink.size}")

            elif entity_type == 'additional_options':
                option = self._map_additional_option(entity_value)
                current_drink.additional_options.append(option)
                logging.debug(f"추가 옵션 추가: {option}")

            # 기타 엔티티 처리 로직 추가 가능

        # 마지막 음료 주문을 완료하고 리스트에 추가
        if current_drink.drink_type:
            self._complete_order(current_drink)

    # 주문이 완성되지 않은 필드 기본값 설정 후 drinks 리스트에 추가 메서드
    def _complete_order(self, order: Drink):
        """
        주문이 완성되지 않은 필드를 기본값으로 설정하고 음료 목록에 추가하는 메서드
        """
        hot_drinks = ["허브티"]  # 항상 핫으로만 제공되는 음료 리스트
        ice_only_drinks = ["토마토주스", "키위주스", "망고스무디", "딸기스무디", "레몬에이드", "복숭아아이스티"]  # 항상 아이스로만 제공되는 음료 리스트

        # 음료 타입에 따른 온도 설정
        if order.drink_type in ice_only_drinks:
            order.temperature = "아이스"
            logging.info(f"{order.drink_type}는(은) 온도를 변경할 수 없습니다. 아이스로 제공됩니다.")
        elif order.drink_type in hot_drinks:
            order.temperature = "핫"
            logging.info(f"{order.drink_type}는(은) 온도를 변경할 수 없습니다. 핫으로 제공됩니다.")
        else:
            order.temperature = order.temperature if order.temperature else "핫"

        if not order.temperature:
            order.temperature = self._set_default_temperature(order.drink_type)
            logging.debug(f"음료 타입에 따른 기본 온도 설정: {order.temperature}")

        # 사이즈 설정 (이미 할당되었을 경우 무시)
        if not order.size:
            order.size = "미디움"
            logging.debug(f"기본 사이즈 설정: {order.size}")

        self.drinks.append(order)
        logging.info(f"{order.drink_type} {order.quantity}잔이 {order.temperature}로 추가되었습니다.")

    # 온도 매핑 알고리즘 메서드
    def _find_previous_or_next_temperature_entity(self, current_index: int) -> str:
        """
        현재 인덱스를 기준으로 이전 또는 다음에 위치한 온도 엔티티를 찾아 반환하는 메서드
        """
        # drink_type의 바로 앞의 엔티티가 temperature인 경우
        if current_index > 0 and self.entities[current_index - 1]['entity'] == 'temperature':
            return self._map_temperature(self.entities[current_index - 1]['value'])

        # drink_type의 바로 뒤에 위치한 엔티티 중 temperature를 찾음
        for i in range(current_index + 1, len(self.entities)):
            if self.entities[i]['entity'] == 'temperature':
                if i + 1 < len(self.entities) and self.entities[i + 1]['entity'] == 'drink_type':
                    # 다음 엔티티가 drink_type인 경우 해당 온도를 무시
                    return None
                return self._map_temperature(self.entities[i]['value'])
        return None

    # 사이즈 매핑 알고리즘 메서드 (추가)
    def _find_next_or_previous_size_entity(self, current_index: int) -> str:
        """
        현재 인덱스를 기준으로 이전 또는 다음에 위치한 사이즈 엔티티를 찾아 반환하는 메서드
        """
        # drink_type의 바로 앞의 엔티티가 size인 경우
        if current_index > 0 and self.entities[current_index - 1]['entity'] == 'size':
            return self._map_size(self.entities[current_index - 1]['value'])

        # drink_type의 바로 뒤에 위치한 엔티티 중 size를 찾음
        for i in range(current_index + 1, len(self.entities)):
            if self.entities[i]['entity'] == 'size':
                if i + 1 < len(self.entities) and self.entities[i + 1]['entity'] == 'drink_type':
                    # 다음 엔티티가 drink_type인 경우 해당 사이즈를 무시
                    return None
                return self._map_size(self.entities[i]['value'])
        return None

    # 온도 값 표준화(아이스, 핫) 후 매핑 메서드
    def _map_temperature(self, value: str) -> str:
        """
        온도 값을 표준화하여 반환하는 메서드
        """
        return standardize_temperature(value)

    def _map_size(self, value: str) -> str:
        """
        사이즈 값을 표준화하여 반환하는 메서드
        """
        return standardize_size(value)

    def _map_additional_option(self, value: str) -> str:
        """
        추가 옵션 값을 표준화하여 반환하는 메서드
        """
        return standardize_option(value)

    def get_mapped_data(self) -> List[Drink]:
        """
        매핑된 음료 데이터 리스트를 반환하는 메서드
        """
        return self.drinks

    def _set_default_temperature(self, drink_type: str) -> str:
        """
        음료 타입에 따른 기본 온도를 설정하는 메서드
        """
        hot_drinks = ["허브티"]  # 항상 핫으로만 제공되는 음료 리스트
        ice_only_drinks = ["토마토주스", "키위주스", "망고스무디", "딸기스무디", "레몬에이드", "복숭아아이스티"]  # 항상 아이스로만 제공되는 음료 리스트
        if drink_type in ice_only_drinks:
            return "아이스"
        elif drink_type in hot_drinks:
            return "핫"
        else:
            return "핫"  # 기본값
    


 
# 주문
class ActionOrderConfirmation(Action):
    def name(self) -> Text:
        return "action_order_confirmation"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # # 현재 주문 정보 초기화(주문을 하는데 이전 주문 정보가 남아있으면 안됨)
            # order_manager.clear_order()
            # 최근 사용자 메시지에서 엔터티를 가져오기
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            user_text = tracker.latest_message.get("text", "")

            if "사이즈 업" in user_text:
                raise KeyError("size up")

            # 엔티티를 위치 순서로 정렬
            mapper = OrderMapper(entities)
            drinks = mapper.get_mapped_data()
        
            logging.warning(f"주문 엔티티: {entities}")
            logging.warning(f"온도, 커피, 사이즈, 잔 수, 옵션: {[(drink.temperature, drink.drink_type, drink.size, drink.quantity, ', '.join(drink.additional_options)) for drink in drinks]}")
            logging.warning(f"Drinks: {drinks}")
            raise_missing_attribute_error(drinks)  # 음료 속성 검증
           

            # 고정된 온도 음료의 온도 확인
            hot_drinks = ["허브티"]
            ice_only_drinks = ["토마토주스", "키위주스", "망고스무디", "딸기스무디", "레몬에이드", "복숭아아이스티"]

            for drink in drinks:
                if drink.drink_type in hot_drinks and drink.temperature != "핫":
                    raise ValueError(f"{drink.drink_type}는(은) 온도를 변경하실 수 없습니다.")
                if drink.drink_type in ice_only_drinks and drink.temperature != "아이스":
                    raise ValueError(f"{drink.drink_type}는(은) 온도를 변경하실 수 없습니다.")

            

            for drink in drinks:
                order_manager.add_order(
                    drink_type=drink.drink_type,
                    quantity=drink.quantity,
                    temperature=drink.temperature,
                    size=drink.size,
                    additional_options=drink.additional_options
                )
            
            confirmation_message = f"주문하신 음료는 {', '.join(order_manager.get_order_summary())}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)

        except ValueError as e:
            dispatcher.utter_message(text=str(e))
        except Exception as e:
            dispatcher.utter_message(text=f"주문 접수 중 오류가 발생했습니다: {str(e)}")
        return []
    
# 주문 변경
class ActionModifyOrder(Action):
    def name(self) -> Text:
        return "action_modify_order"

    # 주어진 텍스트 범위 내에서 엔티티 추출
    def extract_entities(self, text, tracker):
        entities = []
        for entity in tracker.latest_message.get("entities", []):
            if entity.get("extractor") != "DIETClassifier" and entity["start"] >= text["start"] and entity["end"] <= text["end"]:
                entities.append(entity)
        return entities

    # 액션 실행 메소드
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 가장 최근 사용자 메시지에서 엔티티 추출
            modify_entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
        
            mapper = OrderMapper(modify_entities)
            drinks = mapper.get_mapped_data()
            user_text = tracker.latest_message.get("text", "")

            logging.warning(f"사용자 주문 변경 입력 내용: {modify_entities}")
            logging.warning(f"사용자 주문 변경 입력 내용: {user_text}")

            raise_missing_attribute_error(drinks)  # 음료 속성 검증

            # '대신' 또는 '말고'를 사용하여 주문 변경 여부 확인
            if "대신" in user_text or "말고" in user_text or "은" in user_text or "는" in user_text:
                # '대신' 또는 '말고'를 기준으로 텍스트 분리
                split_text = re.split("대신|말고|은|는", user_text)
                logging.warning(f"매칭 시도: {split_text}")
                if len(split_text) == 2:
                    target_part = {"text": split_text[0].strip(), "start": 0, "end": len(split_text[0].strip())}
                    new_part = {"text": split_text[1].strip(), "start": len(split_text[0]) + 2, "end": len(user_text)}

                    logging.warning(f"target_part 내용: {target_part}")
                    logging.warning(f"new_part 내용: {new_part}")

                    # 각각의 텍스트 부분에서 엔티티 추출
                    target_entities = self.extract_entities(target_part, tracker)
                    new_entities = self.extract_entities(new_part, tracker)

                    logging.warning(f"target_entities 내용: {target_entities}")
                    logging.warning(f"new_entities 내용: {new_entities}")

                    # 대상 및 새 엔티티를 매핑하여 데이터 추출
                    target_mapper = OrderMapper(target_entities)
                    target_drinks = target_mapper.get_mapped_data()

                    new_mapper = OrderMapper(new_entities)
                    new_drinks = new_mapper.get_mapped_data()

                    logging.warning(f"target_mapper 내용: {[(drink.temperature, drink.drink_type, drink.size, drink.quantity, ', '.join(drink.additional_options)) for drink in target_drinks]}")
                    logging.warning(f"new_mapper 내용: {[(drink.temperature, drink.drink_type, drink.size, drink.quantity, ', '.join(drink.additional_options)) for drink in new_drinks]}")

                    # 고정된 온도 음료의 온도 확인
                    hot_drinks = ["허브티"]
                    ice_only_drinks = ["토마토주스", "키위주스", "망고스무디", "딸기스무디", "레몬에이드", "복숭아아이스티"]

                    for drink in new_drinks:
                        if drink.drink_type in hot_drinks and drink.temperature != "핫":
                            raise ValueError(f"{drink.drink_type}는(은) 온도를 변경하실 수 없습니다.")
                        if drink.drink_type in ice_only_drinks and drink.temperature != "아이스":
                            raise ValueError(f"{drink.drink_type}는(은) 온도를 변경하실 수 없습니다.")

                    # 기존 주문에서 대상 항목 제거, 제거할 음료, 수량, 온도, 사이즈, 추가 옵션을 가져옵니다.
                    for drink in target_drinks:
                        existing_order = order_manager.find_order(drink.drink_type, drink.temperature, drink.size)
                        if existing_order:
                            order_manager.subtract_order(existing_order.order_id)
                        else:
                            raise ValueError(f"{drink.drink_type}에 대한 주문을 찾을 수 없습니다.")

                    # 새 항목을 기존 주문에 추가, 추가할 음료, 수량, 온도, 사이즈, 추가 옵션을 가져옵니다.
                    for drink in new_drinks:
                        if drink.drink_type in order_manager.ice_only_drinks:
                            dispatcher.utter_message(text=f"{drink.drink_type}는 아이스만 가능합니다.")
                            drink.temperature = "아이스"
                        elif drink.drink_type in order_manager.hot_drinks:
                            dispatcher.utter_message(text=f"{drink.drink_type}는 핫만 가능합니다.")
                            drink.temperature = "핫"
                        order_manager.add_order(
                            drink_type=drink.drink_type,
                            quantity=drink.quantity,
                            temperature=drink.temperature,
                            size=drink.size,
                            additional_options=drink.additional_options
                        )
            else:
                order_manager.clear_order()

                logging.warning(f"주문 변경 엔티티: {modify_entities}")

                for drink in drinks:
                    if drink.drink_type in order_manager.ice_only_drinks:
                        dispatcher.utter_message(text=f"{drink.drink_type}는 아이스만 가능합니다.")
                        drink.temperature = "아이스"
                    elif drink.drink_type in order_manager.hot_drinks:
                        dispatcher.utter_message(text=f"{drink.drink_type}는 핫만 가능합니다.")
                        drink.temperature = "핫"    
                    order_manager.add_order(
                        drink_type=drink.drink_type,
                        quantity=drink.quantity,
                        temperature=drink.temperature,
                        size=drink.size,
                        additional_options=drink.additional_options
                    )

            order_summary = order_manager.get_order_summary()
            summary_list = ', '.join(order_summary)
            confirmation_message = f"주문이 수정되었습니다. 현재 주문은 {summary_list}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)
            return []
        except Exception as e:
            dispatcher.utter_message(text=f"주문 변경 중 오류가 발생했습니다: {str(e)}")
            return []

# 주문 제거
class ActionSubtractFromOrder(Action):
    def name(self) -> Text:
        # 액션의 이름을 반환하는 메소드
        return "action_subtract_from_order"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 최근 사용자 메시지에서 DIETClassifier가 아닌 엔티티들을 가져옴
            subtract_entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            
            # OrderMapper를 사용하여 엔티티들을 매핑
            mapper = OrderMapper(subtract_entities)
            drinks = mapper.get_mapped_data()

            logging.warning(f"사용자 주문 제거 입력 내용: {subtract_entities}")
            logging.warning(f"사용자 주문 제거 매핑 데이터: {[(drink.temperature, drink.drink_type, drink.size, 1, ', '.join(drink.additional_options)) for drink in drinks]}")

            raise_missing_attribute_error(drinks)  # 음료 속성 검증(아이스 들어간 음료 다 취소. 가 드링크 타입이 없는데 왜 에러가 안뜨지?)

            # 매핑된 데이터를 순회하면서 주문 제거
            for drink in drinks:
                existing_order = order_manager.find_order(drink.drink_type, drink.temperature, drink.size)
                if existing_order:
                    order_manager.subtract_order(existing_order.order_id)
                    dispatcher.utter_message(text=f"{drink.drink_type} {drink.quantity}잔이 주문에서 제거되었습니다.")
                else:
                    dispatcher.utter_message(text=f"{drink.drink_type}은(는) 현재 주문에 없습니다.")
            order_summary = ', '.join(order_manager.get_order_summary())
            confirmation_message = f"현재 주문은 {order_summary}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)

            return []
        except Exception as e:
            logging.exception("Exception occurred in action_subtract_from_order")
            dispatcher.utter_message(text=f"주문 제거 중 오류가 발생했습니다: {str(e)}")
            return []
        
# 주문 다중처리
class ActionAddSubtract(Action):
    def name(self) -> Text:
        return "action_add_subtract"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 최근 사용자 메시지에서 엔티티 가져오기
            entities = sorted([entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"], key=lambda x: x['start'])

            add_entities = []
            subtract_entities = []

            current_drink = Drink()
            # 추가하는 부분과 제거하는 부분을 나누기 위한 값을 저장.
            current_action = None

            logging.warning(f"주문 다중처리 엔티티: {entities}")
            
            # 엔티티를 인덱스 값과 같이 가장 처음부터 순서대로 반복합니다.
            for i, entity in enumerate(entities):
                if entity['entity'] == 'add' and (i == 0 or entities[i-1]['entity'] != 'additional_options'):
                    current_action = 'add'
                    if current_drink.drink_type:
                        add_entities.append(current_drink)
                    current_drink = Drink()
                    logging.warning(f"다중처리 추가 : {add_entities}")
                elif entity['entity'] == 'subtract':
                    current_action = 'subtract'
                    if current_drink.drink_type:
                        subtract_entities.append(current_drink)
                    current_drink = Drink()
                    logging.warning(f"다중처리 제거 : {subtract_entities}")
                else:
                    self._map_entity_to_drink(entity, current_drink)

            # 누락 주문 처리(다중처리 인텐트가 인식되서 기능은 실행이 되었는데 채팅이 ~는 추가해주시고 나 ~는 빼주시고 에서 말이 끊기거나 했을 때 실행되도록)
            if current_drink.drink_type:
                if current_action == 'add':
                    add_entities.append(current_drink)
                elif current_action == 'subtract':
                    subtract_entities.append(current_drink)

            # 온도의 기본값을 매핑
            self._set_default_temperature(add_entities)
            self._set_default_temperature(subtract_entities)

            # 매핑된 데이터 출력
            logging.warning(f"추가 엔티티: {add_entities}, 제거 엔티티: {subtract_entities}")

            # 추가 엔티티가 있는 경우 처리
            for drink in add_entities:
                self._process_add(drink)

            # 제거 엔티티가 있는 경우 처리
            for drink in subtract_entities:
                self._process_subtract(drink, dispatcher)

            # 정리된 최종 주문 리스트를 생성
            order_summary = order_manager.get_order_summary()
            summary_list = ', '.join(order_summary)
            confirmation_message = f"주문이 수정되었습니다. 현재 주문은 {summary_list}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)
            return []
        except Exception as e:
            logging.exception("Exception occurred in action_add_subtract")
            dispatcher.utter_message(text=str(e))
            return []

    # 음료의 종류와 온도를 제외한 기본값 메서드
    def _initialize_order(self):
        drink = Drink()
        drink.size = "미디움"
        drink.quantity = 1
        drink.additional_options = []
        return drink

    # 음료 매핑 메서드
    def _map_entity_to_drink(self, entity, drink):
        if entity['entity'] == 'drink_type':
            if entity['value'] in ["아아", "아 아", "아", "아가"]:
                drink.drink_type = "아메리카노"
                drink.temperature = "아이스"
            
            elif entity['value'] in ["뜨아", "뜨 아", "뜨아아", "또", "응아", "쁘허", "뚜아"]:
                drink.drink_type = "아메리카노"
                drink.temperature = "핫"
            elif entity['value'] in ["아카라", "아까라"]:
                drink.temperature = "아이스"
                drink.drink_type = "카페라떼"
            elif entity['value'] in ["아샷추", "아샤추", "아샷 추", "아샸츄","아사츄","아샤츄", "아사추"]:
                drink.additional_options = ["샷"]
                drink.drink_type = "복숭아아이스티"
            else:
                drink.drink_type = standardize_drink_name(entity['value'])
        elif entity['entity'] == 'quantity':
            quantity = entity['value']
            drink.quantity = int(quantity) if quantity.isdigit() else korean_to_number(quantity)
        elif entity['entity'] == 'temperature':
            if drink.temperature is None:
                drink.temperature = standardize_temperature(entity['value'])
        elif entity['entity'] == 'size':
            drink.size = standardize_size(entity['value'])
        elif entity['entity'] == 'additional_options':
            drink.additional_options.append(standardize_option(entity['value']))

    # 온도 기본값 성정 메서드

    def _set_default_temperature(self, drinks):
        hot_drinks = ["허브티"]
        ice_only_drinks = ["토마토주스", "키위주스", "망고스무디", "딸기스무디", "레몬에이드", "복숭아아이스티"]

        for drink in drinks:
            if drink.temperature is None:
                if drink.drink_type in ice_only_drinks:
                    drink.temperature = "아이스"
                elif drink.drink_type in hot_drinks:
                    drink.temperature = "핫"
                else:
                    drink.temperature = "핫"
            else:
                if drink.drink_type in hot_drinks and drink.temperature != "핫":
                    raise ValueError(f"{drink.drink_type}는(은) 온도가 핫으로 고정된 음료입니다! 다시 주문해 주세요.")
                elif drink.drink_type in ice_only_drinks and drink.temperature != "아이스":
                    raise ValueError(f"{drink.drink_type}는(은) 온도가 아이스로 고정된 음료입니다! 다시 주문해 주세요.")

    # 음료 제거 메서드
    def _process_subtract(self, drink, dispatcher):
        try:
            existing_order = order_manager.find_order(drink.drink_type, drink.temperature, drink.size)
            if existing_order:
                order_manager.subtract_order(existing_order.order_id)
            else:
                raise ValueError(f"{drink.drink_type}은(는) 등록되지 않은 커피입니다! 다시 주문해주세요.")
        except ValueError as e:
            dispatcher.utter_message(text=str(e))

    # 음료 추가 메서드
    def _process_add(self, drink):
        order_manager.add_order(
            drink_type=drink.drink_type,
            quantity=drink.quantity,
            temperature=drink.temperature,
            size=drink.size,
            additional_options=drink.additional_options
        )
       
# 주문 확인
class ActionOrderFinish(Action):
    def name(self) -> Text:
        return "action_order_finish"

    # 주문을 완료하는 액션 실행
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            orders = order_manager.get_orders()
            if not orders:
                dispatcher.utter_message(text="주문하신 음료가 없습니다. 음료를 주문해 주세요.")
                return []

            confirmation_message = f"주문하신 음료는 {', '.join(order_manager.get_order_summary())}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)
            
            return []
        except Exception as e:
            dispatcher.utter_message(text=f"주문 완료 중 오류가 발생했습니다: {str(e)}")
            return []

# 주문 취소
class ActionCancelOrder(Action):
    def name(self) -> Text:
        return "action_cancel_order"
    
    #내가 추가한 부분
    def run(self, dispatcher, tracker, domain):
        drink_type = tracker.get_slot('drink_type')  # 음료 이름 추출
        if drink_type in self.orders:
            # 음료 취소 처리
            self.orders[drink_type] -= 1  # 해당 음료 수량 감소
            if self.orders[drink_type] <= 0:
                del self.orders[drink_type]  # 음료 수량이 0 이하이면 주문 목록에서 삭제
            dispatcher.utter_message(f"{drink_type}가 주문에서 제거되었습니다.")
        else:
            dispatcher.utter_message(f"{drink_type}은(는) 현재 주문에 없습니다.")
        return []
    #여기까지

    # 주문을 취소하는 액션 실행
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 현재 주문된 음료 목록 가져오기
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            
            mapper = OrderMapper(entities)
            drinks = mapper.get_mapped_data()

            logging.warning(f"주문 취소 엔티티: {entities}")
            logging.warning(f"주문 취소 매핑 데이터: {[(drink.temperature, drink.drink_type, drink.size, drink.quantity, ', '.join(drink.additional_options)) for drink in drinks]}")

            if drinks:
                for drink in drinks:
                    existing_order = order_manager.find_order(drink.drink_type, drink.temperature, drink.size)
                    if existing_order:
                        order_manager.subtract_order(existing_order.order_id)
                dispatcher.utter_message(text=f"{', '.join([drink.drink_type for drink in drinks])} 주문이 취소되었습니다.")
            else:
                order_manager.clear_order()
                dispatcher.utter_message(text="주문이 취소되었습니다.")

            return []
        except Exception as e:
            dispatcher.utter_message(text=f"주문 취소 중 오류가 발생했습니다: {str(e)}")
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
        # recommedded_message = f"{recommedded_text} {recommended_coffees_str} 등이 있습니다. 어떤 것을 원하시나요?"
        dispatcher.utter_message(text=recommedded_message)

        return []

# 커피 사이즈 변경
class ActionSelectCoffeeSize(Action):
    def name(self) -> Text:
        return "action_select_coffee_size"  # 액션의 이름을 정의하는 메서드

    # start 값이 가장 큰, 즉 사용자의 대화에서 가장 마지막에 오는 size 엔티티 값을 추출
    def extract_last_size(self, entities):
        # size 엔티티를 필터링하여 해당 엔티티들만 리스트로 만듦
        size_entities = [entity for entity in entities if entity["entity"] == "size"]
        # 필터링된 size 엔티티가 있을 경우
        if size_entities:
            # 'start' 값을 기준으로 가장 큰 엔티티를 찾고, 해당 엔티티의 'value' 값을 반환
            return max(size_entities, key=lambda x: x['start'])["value"]
        return None  # size 엔티티가 없을 경우 None 반환

    # 커피 사이즈를 변경하는 액션 실행
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            # 최근 사용자 메시지에서 엔터티를 가져오기
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            
            mapper = OrderMapper(entities)
            drinks = mapper.get_mapped_data()

            logging.warning(f"사이즈 변경 엔티티: {entities}")
            logging.warning(f"사이즈 변경 매핑 데이터: {[(drink.temperature, drink.drink_type, drink.size, drink.quantity, ', '.join(drink.additional_options)) for drink in drinks]}")
            if not drinks:
                raise ValueError("주문 정보를 찾을 수 없습니다.")

            # 엔티티를 위치 순서로 정렬하고 매핑
            mapper = OrderMapper(entities, is_size_change=True)
            temperatures, drink_types, sizes, quantities, additional_options = mapper.get_mapped_data()

            drink = drinks[0]
            new_size = drink.size

            existing_order = order_manager.find_order(drink.drink_type, drink.temperature, None)
            if existing_order:
                order_manager.subtract_order(existing_order.order_id)
                existing_order.size = new_size
                order_manager.add_order(
                    drink_type=existing_order.drink_type,
                    quantity=existing_order.quantity,
                    temperature=existing_order.temperature,
                    size=existing_order.size,
                    additional_options=existing_order.additional_options
                )
                dispatcher.utter_message(text=f"{drink.drink_type}의 사이즈가 {new_size}로 변경되었습니다.")
            else:
                raise ValueError(f"{drink.drink_type}에 대한 주문을 찾을 수 없습니다.")

            order_summary = ', '.join(order_manager.get_order_summary())
            confirmation_message = f"현재 주문은 {order_summary}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)

            return []
        except ValueError as e:
            dispatcher.utter_message(text=str(e))
        except Exception as e:
            dispatcher.utter_message(text=f" 사이즈 변경 중 오류가 발생했습니다: {str(e)}")
            return []

# 커피 온도 변경
class ActionSelectCoffeeTemperature(Action):
    def name(self) -> Text:
        return "action_select_coffee_temperature"  # 액션의 이름을 정의하는 메서드

    # 가장 마지막에 오는 temperature 엔티티 값을 추출하고 변환
    def extract_last_temperature(self, entities):
        # temperature 엔티티를 필터링하여 해당 엔티티들만 리스트로 만듦
        temperature_entities = [entity for entity in entities if entity["entity"] == "temperature"]
        if temperature_entities:
            last_temp = max(temperature_entities, key=lambda x: x['start'])["value"]
            # '차갑게', '시원하게', '뜨겁게', '따뜻하게'를 각각 '아이스'와 '핫'으로 변환
            if last_temp in ["차갑게", "시원하게", "차가운", "시원한"]:
                return "아이스"
            elif last_temp in ["뜨겁게", "따뜻하게", "뜨거운", "따뜻한", "뜨뜻한", "따듯한"]:
                return "핫"
            return standardize_temperature(last_temp)
        return None  # temperature 엔티티가 없을 경우 None 반환

    # 커피 온도를 변경하는 액션 실행
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            
            mapper = OrderMapper(entities)
            drinks = mapper.get_mapped_data()

            logging.warning(f"온도 변경 엔티티: {entities}")
            logging.warning(f"온도 변경 매핑 데이터: {[(drink.temperature, drink.drink_type, drink.size, drink.quantity, ', '.join(drink.additional_options)) for drink in drinks]}")

            if not drinks:
                raise ValueError("주문 정보를 찾을 수 없습니다.")

            drink = drinks[0]
            new_temperature = drink.temperature

            hot_drinks = ["허브티"]
            ice_only_drinks = ["토마토주스", "키위주스", "망고스무디", "딸기스무디", "레몬에이드", "복숭아아이스티"]

            if drink.drink_type in hot_drinks and new_temperature == "아이스":
                raise ValueError(f"{drink.drink_type}는(은) 아이스로 변경할 수 없습니다.")
            if drink.drink_type in ice_only_drinks and new_temperature == "핫":
                raise ValueError(f"{drink.drink_type}는(은) 핫으로 변경할 수 없습니다.")

            existing_order = order_manager.find_order(drink.drink_type, None, drink.size)
            if existing_order:
                order_manager.subtract_order(existing_order.order_id)
                existing_order.temperature = new_temperature
                order_manager.add_order(
                    drink_type=existing_order.drink_type,
                    quantity=existing_order.quantity,
                    temperature=existing_order.temperature,
                    size=existing_order.size,
                    additional_options=existing_order.additional_options
                )
                dispatcher.utter_message(text=f"{drink.drink_type}의 온도가 {new_temperature}로 변경되었습니다.")
            else:
                raise ValueError(f"{drink.drink_type}에 대한 주문을 찾을 수 없습니다.")

            order_summary = ', '.join(order_manager.get_order_summary())
            confirmation_message = f"현재 주문은 {order_summary}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)

            return []
        except ValueError as e:
            dispatcher.utter_message(text=str(e))
        except Exception as e:
            dispatcher.utter_message(text=f"온도 변경 중 오류가 발생했습니다: {str(e)}")
            return []

# 커피 추가옵션 추가
class ActionAddAdditionalOption(Action):
    def name(self) -> Text:
        return "action_add_additional_option"

    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            
            mapper = OrderMapper(entities)
            drinks = mapper.get_mapped_data()

            logging.warning(f"추가 옵션 추가 입력 내용: {entities}")
            logging.warning(f"추가 옵션 추가 매핑 데이터: {[(drink.temperature, drink.drink_type, drink.size, drink.quantity, ', '.join(drink.additional_options)) for drink in drinks]}")

            if not drinks:
                raise ValueError("추가할 음료를 찾을 수 없습니다.")

            drink = drinks[0]
            existing_order = order_manager.find_orders(
                drink_type=drink.drink_type,
                temperature=drink.temperature,
                size=drink.size
            )

            if not existing_order:
                # 기존 주문이 없으면 새로운 주문 추가
                order_manager.add_order(
                    drink_type=drink.drink_type,
                    quantity=drink.quantity,
                    temperature=drink.temperature,
                    size=drink.size,
                    additional_options=drink.additional_options
                )
                logging.info(f"{drink.temperature} {drink.drink_type} {drink.quantity}잔이 추가되었습니다.")
                dispatcher.utter_message(text=f"{drink.temperature} {drink.drink_type} {drink.quantity}잔이 추가되었습니다.")
            else:
                # 기존 주문에 추가 옵션 추가
                for order in existing_order:
                    for option in drink.additional_options:
                        if option not in order.additional_options:
                            order.additional_options.append(option)
                            logging.info(f"{order.drink_type}에 {option} 옵션이 추가되었습니다.")
                order_manager.update_order(existing_order[0])  # 첫 번째 매칭되는 주문 업데이트

                option_str = ', '.join(drink.additional_options)
                dispatcher.utter_message(text=f"{drink.drink_type}에 {option_str} 옵션이 추가되었습니다.")

            # 현재 주문 요약 출력
            order_summary = ', '.join(order_manager.get_order_summary())
            confirmation_message = f"현재 주문은 {order_summary}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)

            return []
        except Exception as e:
            logging.exception("Exception occurred in action_add_additional_option")
            dispatcher.utter_message(text=f"추가 옵션 추가 중 오류가 발생했습니다: {str(e)}")
            return []

# 커피 추가옵션 제거        
class ActionRemoveAdditionalOption(Action):
    def name(self) -> Text:
        return "action_remove_additional_option"
    
    async def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        try:
            entities = [entity for entity in tracker.latest_message.get("entities", []) if entity.get("extractor") != "DIETClassifier"]
            
            mapper = OrderMapper(entities)
            drinks = mapper.get_mapped_data()
            
            logging.warning(f"추가 옵션 제거 입력 내용: {entities}")
            logging.warning(f"추가 옵션 제거 매핑 데이터: {[(drink.temperature, drink.drink_type, drink.size, drink.quantity, ', '.join(drink.additional_options)) for drink in drinks]}")

            current_orders = order_manager.get_orders()

            if len(current_orders) == 1 and not drinks:
                drink = list(current_orders.values())[0]
            elif drinks:
                drink = drinks[0]
            else:
                raise ValueError("주문 정보를 찾을 수 없습니다.")
            


            if drink.additional_options:
                last_remove_option = drink.additional_options[-1]
                drink.additional_options.pop()
                order_manager.update_order(drink)
                dispatcher.utter_message(text=f"{last_remove_option} 옵션이 제거되었습니다.")
            else:
                dispatcher.utter_message(text="제거할 추가 옵션이 없습니다.")

            order_summary = ', '.join(order_manager.get_order_summary())
            confirmation_message = f"현재 주문은 {order_summary}입니다. 다른 추가 옵션이 필요하신가요?"
            dispatcher.utter_message(text=confirmation_message)

            return []
        except ValueError as e:
            dispatcher.utter_message(text=str(e))
        except Exception as e:
            dispatcher.utter_message(text=f"추가 옵션 제거 중 오류가 발생했습니다: {str(e)}")
            return []

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
                takeout = standardize_take(last_take_value)

                final_message = f"{takeout} 주문이 완료되었습니다. 결제는 하단의 카드리더기로 결제해 주시기 바랍니다. 감사합니다."
                dispatcher.utter_message(text=final_message)
            else:
                dispatcher.utter_message(text="테이크아웃 여부를 확인할 수 없습니다.")

            return []
        except Exception as e:
            logging.exception("Exception occurred in action_takeout")
            dispatcher.utter_message(text="주문 완료 중 오류가 발생했습니다. 다시 시도해주세요.")
            return []
        
""""
# "아아 빼주세요"라고 했을 때 전체가 취소되는 문제를 해결한 코드(actions)
class ActionCancelSpecificOrder(Action):
    def name(self):
        return "action_cancel_specific_order"

    def run(self, dispatcher, tracker, domain):
        drink_type = tracker.get_slot('drink_type')  # 슬롯에서 음료 이름 추출
        if drink_type:
            if drink_type in self.orders:
                self.orders[drink_type] = 0
                dispatcher.utter_message(f"{drink_type}가 취소되었습니다.")
            else:
                dispatcher.utter_message(f"{drink_type}는 현재 주문에 없습니다.")
        else:
            dispatcher.utter_message("취소할 음료의 이름을 말씀해 주세요.")
"""

class Person:
    species = "Homo sapiens"
    
    def __init__(self, name):
        self.name = name
    
    # @classmethod 사용
    @classmethod
    def with_classmethod(cls):
        print(f"클래스: {cls}")
        print(f"종: {cls.species}")
        return cls("홍길동")  # 올바른 클래스로 인스턴스 생성
    
    # @classmethod 없이 일반 메서드
    def without_classmethod(self):
        print(f"인스턴스: {self}")
        print(f"종: {self.species}")
        return Person("홍길동")  # 항상 Person 클래스로만 생성

# 테스트
# person = Person("김철수")

# @classmethod는 클래스에서 직접 호출 가능
result1 = Person.with_classmethod()
# print(f"결과: {result1.name}")  # "홍길동"

# # 일반 메서드는 클래스에서 직접 호출하면 에러!
# try:
#     result2 = Person.without_classmethod()  # 에러 발생!
# except TypeError as e:
#     print(f"에러: {e}")
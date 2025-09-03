"""
Python 매직 메서드(Magic Methods) 완벽 가이드
매직 메서드는 __method__ 형태로 정의되며, Python의 내장 연산자와 함수의 동작을 사용자 정의 클래스에서 구현할 수 있게 해줍니다.
"""

# ===== 1. 기본 객체 메서드 =====

class Person:
    def __init__(self, name, age):
        """객체 생성 시 호출되는 생성자"""
        self.name = name
        self.age = age
    
    def __str__(self):
        """str(obj) 또는 print(obj) 시 호출 - 사용자 친화적 문자열"""
        return f"Person(name={self.name}, age={self.age})"
    
    def __repr__(self):
        """repr(obj) 또는 개발자 모드에서 호출 - 개발자용 문자열"""
        return f"Person('{self.name}', {self.age})"
    
    def __len__(self):
        """len(obj) 호출 시 실행"""
        return len(self.name)
    
    def __bool__(self):
        """bool(obj) 또는 if obj: 조건문에서 호출"""
        return self.age > 0
    
    def aa(self):
        return self.age>0

# 사용 예시
person = Person("Alice", 25)
print(person)           # Person(name=Alice, age=25)
print(repr(person))     # Person('Alice', 25)
print(len(person))      # 5 (이름 길이)
print(bool(person))     # True
print(person.aa)


print('# ===== 2. 컨테이너 메서드 (가장 중요!) =====')

class CustomList:
    def __init__(self, items=None):
        self.items = items or []
    
    def __getitem__(self, index):
        """obj[index] 호출 시 실행 - 인덱싱/슬라이싱"""
        if isinstance(index, slice):
            return CustomList(self.items[index])
        return self.items[index]
    
    def __setitem__(self, index, value):
        """obj[index] = value 호출 시 실행"""
        self.items[index] = value
    
    def __delitem__(self, index):
        """del obj[index] 호출 시 실행"""
        del self.items[index]
    
    def __len__(self):
        """len(obj) 호출 시 실행"""
        return len(self.items)
    
    def __contains__(self, item):
        """item in obj 호출 시 실행"""
        return item in self.items
    
    def __iter__(self):
        """for item in obj 호출 시 실행 - 이터레이터 반환"""
        return iter(self.items)
    
    def __str__(self):
        return f"CustomList({self.items})"

# 사용 예시
custom_list = CustomList([1, 2, 3, 4, 5])
print(custom_list[0])        # 1 (__getitem__)
print(custom_list[1:3])      # CustomList([2, 3]) (__getitem__ with slice)
custom_list[0] = 10          # __setitem__
print(custom_list)           # CustomList([10, 2, 3, 4, 5])
print(len(custom_list))      # 5 (__len__)
print(3 in custom_list)      # True (__contains__)
for item in custom_list:     # __iter__
    print(item, end=" ")     # 10 2 3 4 5

# ===== 3. 산술 연산 메서드 =====

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """obj + other"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """obj - other"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """obj * scalar"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        """obj / scalar"""
        return Vector(self.x / scalar, self.y / scalar)
    
    def __eq__(self, other):
        """obj == other"""
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        """obj < other"""
        return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)
    
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

# 사용 예시
v1 = Vector(2, 3)
v2 = Vector(1, 4)
print(v1 + v2)    # Vector(3, 7)
print(v1 - v2)    # Vector(1, -1)
print(v1 * 2)     # Vector(4, 6)
print(v1 / 2)     # Vector(1.0, 1.5)
print(v1 == v2)   # False
print(v1 < v2)    # True

# ===== 4. 컨텍스트 매니저 메서드 =====

class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """with 문 시작 시 호출"""
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 문 종료 시 호출"""
        print(f"Closing file: {self.filename}")
        if self.file:
            self.file.close()
        # 예외 처리: False 반환 시 예외 재발생, True 반환 시 예외 억제
        return False

# 사용 예시
with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")
# 자동으로 파일이 닫힘

# ===== 5. 호출 가능 객체 메서드 =====

class Multiplier:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, value):
        """obj(args) 호출 시 실행 - 객체를 함수처럼 호출"""
        return value * self.factor

# 사용 예시
double = Multiplier(2)
print(double(5))    # 10 (객체를 함수처럼 호출)
triple = Multiplier(3)
print(triple(4))    # 12

# ===== 6. 속성 접근 메서드 =====

class DynamicObject:
    def __init__(self):
        self._data = {}
    
    def __getattr__(self, name):
        """존재하지 않는 속성에 접근할 때 호출"""
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """속성 설정 시 호출 (obj.attr = value)"""
        if name.startswith('_'):
            # private 속성은 직접 설정
            super().__setattr__(name, value)
        else:
            # public 속성은 _data에 저장
            if not hasattr(self, '_data'):
                super().__setattr__('_data', {})
            self._data[name] = value
    
    def __delattr__(self, name):
        """속성 삭제 시 호출 (del obj.attr)"""
        if name in self._data:
            del self._data[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

# 사용 예시
obj = DynamicObject()
obj.name = "Alice"      # __setattr__
obj.age = 25           # __setattr__
print(obj.name)        # Alice (__getattr__)
print(obj.age)         # 25 (__getattr__)
del obj.age            # __delattr__

# ===== 7. PyTorch Dataset 실제 구현 예시 =====

import torch
from torch.utils.data import Dataset
import numpy as np

class CustomImageDataset(Dataset):
    """실제 PyTorch Dataset에서 매직 메서드 활용 예시"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """데이터셋 크기 반환 - DataLoader에서 사용"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """인덱스로 데이터 접근 - DataLoader에서 배치 생성 시 사용"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 이미지 로드 (실제로는 PIL, OpenCV 등 사용)
        image = np.random.rand(224, 224, 3)  # 더미 이미지
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 사용 예시
dataset = CustomImageDataset(
    image_paths=["img1.jpg", "img2.jpg", "img3.jpg"],
    labels=[0, 1, 0]
)

print(len(dataset))         # 3 (__len__)
image, label = dataset[0]   # __getitem__
print(f"Image shape: {image.shape}, Label: {label}")

# DataLoader에서 자동으로 __getitem__과 __len__ 사용
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch_images, batch_labels in dataloader:
    print(f"Batch images shape: {batch_images.shape}")
    print(f"Batch labels: {batch_labels}")
    break

# ===== 8. 전체 매직 메서드 요약 =====

class CompleteExample:
    """모든 주요 매직 메서드를 포함한 완전한 예시"""
    
    def __init__(self, value):
        self.value = value
    
    # 문자열 표현
    def __str__(self): return f"CompleteExample({self.value})"
    def __repr__(self): return f"CompleteExample({self.value!r})"
    
    # 컨테이너 메서드
    def __len__(self): return len(str(self.value))
    def __getitem__(self, key): return str(self.value)[key]
    def __contains__(self, item): return item in str(self.value)
    
    # 산술 연산
    def __add__(self, other): return CompleteExample(self.value + other.value)
    def __mul__(self, other): return CompleteExample(self.value * other)
    
    # 비교 연산
    def __eq__(self, other): return self.value == other.value
    def __lt__(self, other): return self.value < other.value
    
    # 호출 가능
    def __call__(self, *args): return f"Called with {args}"
    
    # 불린 값
    def __bool__(self): return bool(self.value)

# 모든 기능 테스트
obj = CompleteExample("Hello")
print(obj)              # CompleteExample(Hello)
print(len(obj))         # 5
print(obj[0])          # H
print('H' in obj)      # True
print(obj + CompleteExample(" World"))  # CompleteExample(Hello World)
print(obj * 2)         # CompleteExample(HelloHello)
print(obj == CompleteExample("Hello"))  # True
print(obj("test"))     # Called with ('test',)
print(bool(obj))       # True

print("\n=== 매직 메서드 활용 완료 ===")
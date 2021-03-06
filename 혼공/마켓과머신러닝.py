# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:17:30 2021

@author: as_th
"""

# 혼자 공부하는 머신러닝 딥러닝

# ch 1-3 마켓과 머신러닝

# 주요 정리
 
# 이진 분류
# =============================================================================
# 머신러닝에서 여러 개의 종류(혹은 클래스(class)라고 부른다.)중 하나를 구별해 내는 문제를
# 분류(classification)라고 부른다.
# 2개의 클래스 중에서 하나를 고르는 문제를 이진 분류(binary classification)라고 한다.
# 파이썬의 클래스와는 다른 의미다.
# =============================================================================

# 특성(feature)
# =============================================================================
# 도미의 길이 25.4cm 무게는 242.0g이고 두 번째 도미는 26.3cm 290.0g 이러한 값들을
# 특성이라고 한다.
# 특성은 데이터의 특징이다.
# =============================================================================

# 산점도
# =============================================================================
# 산점도는 x,y축으로 이뤄진 좌표계에 두 변수(x,y)의 관계를 표현하는 방법이다.
# =============================================================================

# 맷플롯립
# =============================================================================
# 파이썬 과학계산용 그래프를 그리는 대표적인 패키
# =============================================================================

# 머신러닝에서의 모델
# =============================================================================
# 머신러닝 알고리즘을 구현한 프로그램을 모델(model)이라고 부른다.
# 또는 프로그램이 아니더라도 알고리즘을 (수식 등으로)구체화하여 표현한 것을 모델이라고 부른다.
# # 예를 들어 "스팸 메일을 걸러내기 위해 k-최근접 이웃 모델을 사용해봅시다"라고 말할 수 있습니다.
# =============================================================================



# 도미 35마리 데이터 
# 생선의 길이와 무게

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]


# 빙어 데이터 준비하기
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]




import matplotlib.pyplot as plt 


plt.scatter(bream_length, bream_weight) 
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length') # x축은 길이
plt.ylabel('weight') # y축은 무게
plt.show()


# 첫 번째 머신러닝 프로그램
# k-최근접 이웃(K-Nearest Neighbors)알고리즘 사용해 도미와 빙어 데이터 구분

length = bream_length + smelt_length #도미 35개의 길이, 빙어 14개의 길이
weight = bream_weight + smelt_weight # 도미 35개의 무게, 빙어 14개의 무게

# zip()함수 이용해 2차원 리스트로 만들기
fish_data = [[l,w] for l, w in zip(length, weight)]
print(fish_data) # [[25.4, 242.0]....] #  [[길이1,무게1],[길이2, 무게2],....]

# 정답 데이터 준비
# 컴퓨터는 문자를 이해하지 못하므로 도미와 빙어를 숫자 1,0으로 표현 도미는1 빙어는 0
# 도미는 35개니 35번 등장, 빙어는 14개니깐 14번 등장

fish_target = [1] * 35 + [0] * 14
print(fish_target)

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier() # 객체 생성

# fish_data와 fish_target을 전달하여 도미를 찾기 위한 기준을 학습
kn.fit(fish_data, fish_target) # 훈련 데이터, 정답 데이터

kn.score(fish_data, fish_target)

# k- 최근접 이웃 알고리즘
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600 ,marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.predict([[30, 600]])

print(kn._fit_X)

print(kn._y)

kn49 = KNeighborsClassifier(n_neighbors=49)

kn49.fit(fish_data, fish_target)
kn49.score(fish_data, fish_target)

print(35/49)




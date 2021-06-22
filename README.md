# 네트워크 초기화

#  네트워크가 최초로 가지고 있어야 할 가중와 편향을 설정

#  보통은 정규분포 랜덤으로 초기화 하거나, 카이밍 히, 사비에르 초깃값 등을 활용

def init_network():

  network = {}

​

  # 1층 매개변수 초기화

  network["W1"] = np.array([[0.1, 0.3, 0.5],

                            [0.2, 0.4, 0.6]])

  network["B1"] = np.array([0.1, 0.2, 0.3])

​

  # 2층 매개변수 초기화

  network["W2"] = np.array([[0.1, 0.4],

                            [0.2, 0.5],

                            [0.3, 0.6]])

  network["B2"] = np.array([0.1, 0.2])

​

  # 3층 매개변수 초기화

  network["W3"] = np.array([[0.1, 0.3],

                            [0.2, 0.4]])

  network["B3"] = np.array([0.1, 0.2])

​

  return network

​

# 순전파 XW+B

def forward(network, x):

  # 가중치, 편향 꺼내기

  W1, W2, W3 = network["W1"], network["W2"], network["W3"]

  b1, b2, b3 = network["B1"], network["B2"], network["B3"]

​

  # 1층 계산

  Z1 = (x @ W1) + b1

  A1 = sigmoid(Z1)

​

  # 2층 계산

  Z2 = A1 @ W2 + b2

  A2 = sigmoid(Z2)

​

  # 3층 계산

  Z3 = A2 @ W3 + b3

  y = identity_function(Z3)

​

  return y

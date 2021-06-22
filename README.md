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

​

def sigmoid(x):

  return 1 / (1 + np.exp(-x))

​

# 테스트 데이터만 가져오기

def get_test_data():

  _, (X_test, t_test) = mnist.load_data()

​

  # YOUR CODE HERE

  # 단순히 X_test로 리턴하는게 아닌 (N, M) 형태로 리턴하기

​

  IMAGE_COUNT = X_test.shape[0] # shape[0] : 이미지개수 (X_test.shape -> (10000,28,28))

  

  X_test_reshaped = X_test.reshape(IMAGE_COUNT, -1)

​

  return X_test_reshaped, t_test

​

# 이미 학습이 완료된 신경망 데이터 가져오기(sample_weight.pkl)

def init_network():

  import pickle

  with open("./sample_weight.pkl", "rb") as f:

    network = pickle.load(f)

  

  return network

​

def predict(network, x):

  # YOUR CODE HERE

  W1, W2, W3 = network["W1"], network["W2"], network["W3"]

  b1, b2, b3 = network["b1"], network["b2"], network["b3"]

​

  # 1. 각 층은 입력되는 값과 해당 층의 가중치의 내적을 구하고 편향을 더한다.

  # 2. (1)에서 계산된 값과 각 층의 활성화 함수를 씌워주고 다음층으로 넘겨준다.

​

  # Layer 1 계산 ( 입력 : x, 가중치 : W1, 편향 : b1, 활성화 함수 : sigmoid, 출력 A1 )

  z1 = x @ W1 + b1

  a1 = sigmoid(z1)

​

  # Layer 2 계산 ( 입력 : a1, 가중치 : W2, 편향 : b2, 활성화 함수 : sigmoid, 출력 A2)

  z2 = a1 @ W2 + b2

  a2 = sigmoid(z2)

​

  # Layer 3 계산 ( 입력 : a2, 가중치 : W3, 편향 : b3, 활성화 함수 : 출력층이기 때문에 softmax, 출력 y)

  z3 = a2 @ W3 + b3

  y = softmax(z3)

​

  # 출력층의 활성화 함수는? softmax를 사용하세여

  return y

​

# 배치를 이용한 예측

X, t = get_test_data()

​

net = init_network()

​

# 배치란? 데이터의 묶음

# batch_size : 1 배치당 들어있어야 하는 데이터의 개수

# 60,000개의 데이터를 batch_size 100으로 묶으면 몇 개의 배치가 만들어질까요? 600개의 배치가 만들어 진다.

​

batch_size = 100

acc_count = 0

​

for i in range(0, len(X), batch_size):

  X_batch = X[i : i + batch_size] # X_batch의 shape : (100, 784)

​

  pred_batch = predict(net, X_batch) # pred_batch의 shape : (100, 10)

  pred_batch = np.argmax(pred_batch, axis=1)

​

  acc_count += np.sum( pred_batch == t[i : i + batch_size])

​

print(float(acc_count) / len(X))

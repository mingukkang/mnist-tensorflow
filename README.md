## MNIST tensorflow code 99.53% (Xavier, batch norm, L2 reg, Learning rate decay)
**이 튜토리얼은 최근에 많이 사용하는 다음과 같은 방법을 사용하였고, 작성자가 모두 직접 코딩을 하여 작성하였습니다.**

1. Xavier_initializer
2. Batch_normalization
3. L2_regularization
4. Learning_rate_decay

**또한, 위의 .py 파일은 Cost(Loss)함수,Learning_rate_decay,텐서플로우 Graph의 모습을 확인하기 위한 tensorboard 관련 코드,**
**saver를 이용하여 check point 파일을 불러오는 예제 코드를 모두 포함하고 있습니다.**
_ _ _
### 1. 작성자 컴퓨터 사양 및 프로그램 버전
**- Cpu : intel i7 -3770**

**- Ram : 16G**

**- GPU : GTX 1080TI(Memory: 11G)**

**- OS: window 10(64bit)**

**- Tensorflow-gpu version:  1.3.0rc2**

**- Tensorflow-gpu version:  1.3.0rc2**
_ _ _
### 2. Code
#### 1.Xavier_initializer
![사진1](https://github.com/MINGUKKANG/mnist_tensorflow/blob/master/images/xavier_initializer_code.PNG)

#### 2.Batch_ normalization
![사진2](https://github.com/MINGUKKANG/mnist_tensorflow/blob/master/images/bach_norm1_code.PNG)
![사진3](https://github.com/MINGUKKANG/mnist_tensorflow/blob/master/images/batch_norm2_code.PNG)

**위의 batch_norm코드의 경우 phase(train or test)를 작성자가 직접 입력하는 것(FLAGS)이므로 최적화를 할 때 따로 위의 두개 이외의 코드를**
**작성할 필요가 없습니다**


#### 3. L2_regularization
![사진4](https://github.com/MINGUKKANG/mnist_tensorflow/blob/master/images/regularization_code.PNG)

#### 4. learning_rate_decay
![사진5](https://github.com/MINGUKKANG/mnist_tensorflow/blob/master/images/learning_rate_decay_code.PNG)
_ _ _
### 3. Model
![사진6](https://github.com/MINGUKKANG/mnist_tensorflow/blob/master/images/model_image.png)
_ _ _
### 4.Train_Accuracy, Cost, Learning_rate
![사진7](https://github.com/MINGUKKANG/mnist_tensorflow/blob/master/images/cost%2Clearning%2Caccur.PNG)
_ _ _
### 5. Test_Accuracy = 99.53%
![사진8](https://github.com/MINGUKKANG/mnist_tensorflow/blob/master/images/Accuracy_test.PNG)

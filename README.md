# Unsupervised representation learning with deep convolutional generative adversarial networks (DCGAN)
## 논문 Review
본 논문에서는 기존의 생성적 deep learning 모델인 generative adversarial network (GAN)을 구성하는데 있어 convolutional neural network (CNN)을 활용하는 방법에 대해 기술한 논문이다.

일반적으로 GAN은 데이터의 확률 분포인 p(x)를 학습하는데 있어 기존의 maximum likelihood 기반의 최적화 기법 대신 deep learning framework에서의 훈련 조건인 cross entorpy를 활용하여 보다 간단하고 편리한 방법으로 p(x)의 학습이 가능하도록 하였고, 이를 통해 특정한  domain의 이미지 또한 데이터를 생성할 수 하나의 방법론을 제시하여 deep learning 연구의 한획을 그은 것으로 평가된다.

GAN은 크게 generator와 discriminator 두개의 part로 나눌 수 있다. Image를 기준으로 설명하면, generator에서는 수백차원의 가우시안 random 잡음을 입력으로 받아 fully connected layer (FCN)를 거친 후, 학습하고자 하는 DB와 동일한 형태의 image를 출력으로 내보내는 것을 목표로 한다. Discriminator에서는 generator에서 생성된 fake image와 실제 학습 DB의 real image를 구분하는 역할을 수행하는 것으로, 1개의 sigmoid outnode를 통해 이를 구분하는 동작을 수행한다.

GAN의 목적은 어디까지나 generator가 random noise를 p(x)에 잘 mapping 시킬 수 있도록 만들어 최대한 random noise 입력이 훈련 DB에 있을법한 image를 생성할 수 있도록 하는데에 있다. 이를 위해 discriminator는 real과 fake를 구분할 수 있도록 학습이 되어야 하며, generator는 discriminator를 잘 **속일 수 있도록** 학습이 이루어져야 한다.

이러한 목적을 위해 GAN에서의 학습은 generator와 discriminator가 분리되어 학습이 이루어진다. Discriminator 학습을 위해서 real image와 fake image에 각각 0과 1값의 label을 할당한 후, 일반적인 classification 문제와 마찬가지로 cross entropy loss를 구하여 gradient **descent**를 수행한다. Generator의 경우 할당된 label과는 반대되는 동작을 수행하도록 할 목적으로 동일한 loss로부터의 gradient **ascent**를 수행한다. 일반적으로 gradient의 부호를 바꾸는 것 보다는, 훈련의 편의상 fake image의 label을 바꾸어 이를 통한 gradient **descent**를 수행하는 것이 일반적인 방법이다.

초기의 GAN은 구현의 편리함으로 인해 FCN 통해 많이 활용되었다. 하지만 이전부터 image classification에서는 CNN이 많은 FCN 대비 많은 강점을 보여왔기에 GAN에도 CNN을 도입하기 위한 많은 노력이 이루어져왔으나, GAN에서 만큼은 CNN을 이용한 구성이 그다지 안정적으로 동작하지 못하였다. 본 논문에서는 많은 시행착오 끝에 GAN에 CNN을 성공적으로 적용할 수 있는 방법을 찾아 그 방법에 대해 자세히 기술하고 있다.
먼저 본 논문에서 언급하고 있는 안정적인 훈련을 위한 조건은 다음과 같다.

1. CNN에서 전통적으로 활용하는 pooling 동작 대신 stride의 변화를 통해서만 layer를 쌓는다.
2. Generator와 discriminator에 batchnorm을 사용한다.
3. Fully connected layer는 사용하지 않는다.
4. Generator에는 ReLU를 Discriminator에는 Leaky ReLU를 사용한다.

이 같은 조건을 통해 DCGAN을 구성하여 진행한 실험에는 다양한 성공적인 결과들을 보여주었다.

# Repository 설명
본 repository에는 Tensorflow 2.0기반으로 DCGAN을 구현한 python

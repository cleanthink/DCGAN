# Unsupervised representation learning with deep convolutional generative adversarial networks (DCGAN)
## 논문 Review
본 논문에서는 기존의 generative deep learning 모델인 generative adversarial network (GAN)을 구성하는데 있어 convolutional neural network (CNN)을 활용하는 방법에 대해 기술하고 있습니다.

GAN은 데이터의 확률 분포인 p(x)를 학습하는데 있어 기존의 maximum likelihood 기반의 최적화 기법 대신 deep learning framework에서의 훈련 조건인 cross entorpy를 활용하여 보다 편리한 방법으로 p(x)의 학습이 가능하도록 하였고, 이를 통해 p(x) 분포 내의 이미지 또는 데이터를 생성할 수 있게 해주는 하나의 방법론을 제시하여 deep learning 연구의 한획을 그은 것으로 평가되고 있습니다.

GAN은 크게 generator와 discriminator 두개의 part로 나눌 수 있습니다. Image를 기준으로 설명하면, generator는 수백차원의 가우시안 random 잡음을 입력으로 받아 fully connected layer (FCN)를 거친 후, 학습하고자 하는 DB와 비슷한 형태의 image를 출력으로 내보내는 것을 목표로 합니다. Discriminator는 generator에서 생성된 fake image와 실제 학습 DB의 real image를 구분하는 역할을 수행하는 것으로, 1개의 sigmoid outnode를 통해 이를 구분하는 동작을 수행합니다.

GAN의 목적은 어디까지나 generator가 random noise p(z)를 p(x)에 잘 mapping 시킬 수 있도록 훈련하여 최대한 random noise 입력이 훈련 DB에 있을법한 image를 생성할 수 있도록 하는데에 있습니다. 이를 위해 discriminator는 real과 fake를 잘 구분할 수 있도록 학습이 되어야 하며, generator는 이러한 discriminator를 꾸준히 잘 **속일 수 있도록** 학습이 이루어져야 합니다.

이러한 목적을 위해 GAN에서는 generator와 discriminator가 분리되어 학습이 이루어집니다. Discriminator 학습을 위해서 real image와 fake image에 각각 0과 1값의 label을 할당한 후, 일반적인 classification 문제와 동일하게 cross entropy loss를 구하여 gradient **descent**를 수행합니다. Generator의 경우 기 할당된 label과는 반대되는 동작을 수행하도록 할 목적으로 동일한 loss로부터의 gradient **acent**를 수행합니다. 일반적으로 gradient의 부호를 바꾸는 것 보다는, 훈련의 편의상 fake image의 label을 바꾸어 이를 통한 gradient **descent**를 수행하는 것이 간편하기에 label만 교체후 loss를 새로 구하고 이를 활용하여 훈련합니다.

초기의 GAN은 안정적인 훈련이 가능한 FCN 통해 많이 구현되었습니다. 이전부터 image classification에서는 CNN이 FCN 대비 많은 강점을 보여왔기에 GAN에도 CNN을 도입하기 위한 많은 노력이 이루어져왔으나, GAN에서 만큼은 CNN을 이용한 구성이 그다지 안정적으로 동작하지 못하였습니다. 본 논문에서는 많은 시행착오 끝에 GAN에 CNN을 성공적으로 적용할 수 있는 방법을 찾아 그 방법에 대해 자세히 기술하고 있습니다.
먼저 본 논문에서 언급하고 있는 안정적인 훈련을 위한 조건들은 다음과 같습니다.

1. CNN에서 전통적으로 활용하는 pooling 동작 대신 stride의 변화를 통해서만 CNN layer를 쌓습니다.
2. Generator와 discriminator에 batchnorm을 사용합니다.
3. Fully connected layer는 사용하지 않습니다.
4. Generator에는 ReLU를 Discriminator에는 Leaky ReLU를 사용합니다.

이같은 조건을 통해 저자들은 CNN을 통한 GAN (DCGAN)을 구현에 성공하였으며, generator를 통해 생성된 여러 image sample들을 통해 제안한 방법이 안정적으로 동작함을 입증하였습니다.

## Repository 설명
본 repository에는 jupyter notebook을 통해 DCGAN을 구현한 단일 ipynb 파일만을 포함하고 있습니다.
실험을 수행한 환경은 다음과 같습니다.
- Window 10
- Python: 3.7.3
- Cuda: 10.0
- CuDNN: 7.6.4 for cuda 10.0
- Tensorflow-gpu: 2.0.0
- **gast: 0.2.2** (최신버전과 TF 2.0이 호환되지 않음)

DCGAN 구현에 관련된 모든 설명 및 결과물들은 ipynb 파일안에 포함되어 있습니다.

**본 DCGAN 구현 code는 tensorflow 공식 DCGAN 예제 및 여러 tutorial code를 참조하여 작성되었습니다.**

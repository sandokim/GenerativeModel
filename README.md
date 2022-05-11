### GAN idea

[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

Diffusion models

#### Maximum Likelihood Estimation (MLE)

PDF: 데이터가 해당 분포로부터 샘플링될 확률

Likelihood: 주어진 파라미터를 이용한 분포가 모집단의 분포일 확률

주어진 데이터를 바탕으로 모집단의 분포와 유사할 확률이 가장 높은 파라미터 찾기

VAE: latent variable(z)

Diffusion Models: Latent variable이 Markov chain

Random process: 확률 변수들의 나열

Markov chain: 이전 시점의 변수에만 영향을 받은 random process

<img src="https://github.com/Hyeseong0317/GAN/blob/main/images/markov chain.PNG" width="40%">

#### Score-based Generative Models (NCSN)
Score-based Generative Models (NCSN) : 랜덤 노이즈에서 시작해 score값을 따라 높은 확률값이 있는 공간에서 데이터 생성

Diffusion Models (DDPM) : 노이즈를 제거하는 과정을 학습해 랜덤 노이즈로부터 데이터 생성

Score-based Generative Modeling with SDEs : SDE라는 구조 내에서 NCSN과 DDPM을 통합

[Score-based Generative Modeling by Diffusion Process](https://arxiv.org/pdf/2011.13456.pdf)

### Langevin Dynamics -> Gradient Ascent를 통해 이미지 분포를 잘 나타내는 subspace를 찾아낸다. Noise가 적은 true 이미지를 찾아낸다.

<img src="https://github.com/Hyeseong0317/Generative-Model/blob/main/images/langevin model.jpg" width="60%">

<img src="https://github.com/Hyeseong0317/Generative-Model/blob/main/images/gradient ascent.png" width="40%">

### Random variable & Random process

Random variable : Time X, 일반적인 함수 -> 확률변수는 불확실한 어떤 사건을 숫자로 모델링하는데 사용 

Random process : Time까지 고려, 주가차트를 예시로 생각하면 됩니다. -> Random process는 불확실한 어떤 신호를 모델링하는데 사용

모든 시간에 대해 신호의 값을 정확히 표현할 수 있으면 그 신호를 결정적 신호(deterministic signal)라 하고, 불확실성이 있어서 정확히 표현할 수 없을 때 그 신호를 랜덤 신호(random signal)라 합니다.

#### Autocorrelation의 의미
• RX(t1,t2)는 랜덤 프로세스 X(t)의 통계적 특성이 t1으로부터 τ = t2 - t1 초 지난 후에 얼마나 유사한지를 나타낸다고 볼 수 있다.

• 다음 그림 (a)와 같이 느리게 변하는 랜덤 프로세스에서는 상당히 큰 값의 에 τ에 대해서도 X(t1)과 X(t2)가 상관성을 가지겠지만, 그림 (b)와 같이 빠르게 변하는 랜덤 프로세스에서는 작은 τ에 대해서도 상관성이 거의 없어진다.

• 따라서 자기상관 함수를 알면 그 랜덤 프로세스의 표본 함수의 파형이 빠르게 변하는지 느리게 변하는지 예측할 수 있으며, 결과적으로 주파수 성분에 대한 정보를 알 수 있게 된다.

• 랜덤 프로세스에 대한 전력스펙트럼 밀도(PSD)는 자기상관 함수의 푸리에 변환으로 주어진다. 

### 에르고드성 (Ergodicity)

5. 에르고딕성 ( 앙상블 평균 = 시간 평균 )

  ㅇ 정상상태 과정(랜덤과정의 통계적 성질이 시간에 따라 변하지 않음) 하에서 
     집단 전체에 대한 시간 평균과 앙상블 평균이 같아지게 되는 성질
     
  에르고드성 잇점

  ㅇ 계의 상태변화를 굳이 시간적으로 따라갈 필요 없이, (시간적 특성)
  
  ㅇ 시간 독립적으로 계의 정상상태과정(Stationary Process) 만을, (통계적 특성)
  
  ㅇ 고려해도 마찬가지가 됨

어떤 소설가가 있다. 이 사람은 동시에 여려가지 작품을 집필한다. 이 사람과 만나 현재 집필 중인 소설들의 내용에 대해 인터뷰를 했더니, 네 가지 소설 모두 한국 근대사를 배경으로 하는 역사소설이었다. 만약, 내가 이러한 정보를 근거로 해당 소설가가 (현재 집필중인 작품들을 제외하고) 앞으로 출판할 20권의 책 중 대다수가 역사물 일 것(혹은, 10년 뒤에 이 사람이 작업중인 네 개 소설도 역사소설 인 것)이라고 추측한다면, 이러한 추측은 꽤 타당하지 않을까?

이와 같이 어떠한 시스템의 시간평균과 앙상블평균(혹은 공간 평균)이 동일한 양상을 보일 때, 우리는 그 시스템의 갖는 어떠한 변환과정이  'ergodic 하다'라거나 'ergodic process 이다'라고 일컬는다.

통계역학에서 이러한 성질은 아주 큰 이점을 가져다 준다. 통계역학은 대부분 매우 큰 계에 대해 다루게 되는데, 이 경우 전체를 앙상블 평균을 내지 않고 시간 평균을 내 그 평균값이 충분히 수렴 했을 때 그 값을 이용할 수 있다.

### GAN idea

[Read-through: Wasserstein GAN](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)

#### Wasserstein distance = kantorovich-Rubinstei = Optimal transport = Earth mover's distance 

[Introduction to the Wasserstein distance](https://www.youtube.com/watch?v=CDiol4LG2Ao)

#### High Dimension VS Low Dimension (Intersection X)

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/high_dimension_intersection.JPG" width="40%">

#### Wasserstein Distance Discrete 

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Wasserstein_distance_discrete.JPG" width="40%" align='left'/>

#### Wasserstein Distance Continuous

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Wasserstein_distance_continuous.JPG" width="50%" align='center'/>

#### Joint distribution Gamma

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/joint_distribution_gamma.JPG" width="80%">

#### Taxanomy of generative models

<img src="https://github.com/Hyeseong0317/GAN/blob/main/images/Taxonomy-of-generative-models-based-on-maximum-likelihood.JPG" width="40%">

Explicit density : Data distribution에 approximate할 확률분포를 명시적으로 정하고 감. 확률분포 ex) Gaussian Distribution(=MSE), Bernoulli Distribution(=Cross-entropy)

Implicit density : 확률분포를 정하지 않고 학습. ex) GAN

### Variational Auto Encoder(VAE) 
### -> loss function = Reconstruction error + KL divergence(Gaussian distribution<->Data Distribution)

Variational Auto Encoder의 KL term은 Gaussian distribution말고는 계산하기가 어렵다. 

Encoder : x, Prior; Gaussian distribution, Decoder; sampled z and data x Maximum Likelihood

### Auto Encoder(AE) -> loss function = Reconstruction error

Encoder : x, Decoder; sampled z and data x Maximum Likelihood

수학적으로보면 variational auto encoder(VAE)와 auto encoder(AE)는 하등 관계가 없다..

### Adversarial Auto Encoder(AAE) 

VAE에서 Prior가 Gaussian distribution이 아니면 우리가 선택한 실제 데이터 분포와 유사하다고 가정한 Gaussian distribution과 실제 Data distribution 사이의 차이인 KL divergence를 계산하기 어렵다. -> AAE는 이 문제를 해결한다, KL divergence를 계산하지 않아도 되는 함수를 써보자.

### VAE 와 GAN 차이

VAE는 Encoder와 Decoder 모두 ELBO를 Maixmize하려하기 때문에 서로 으쌰으쌰하면서 학습이 잘되는 반면, GAN은 Generator는 Objective function을 Minimize, Discriminator는 Objective function을 Maximize하려하기 때문에 적대적으로 학습하며 학습이 잘 안된다.

#### Metrices

[Inception scores; Improved techniques for training gans](https://arxiv.org/pdf/1606.03498.pdf)

[FID Scores; GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500.pdf)

### Diffusion models

[Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

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

[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600.pdf)

Since Langevin dynamics use ∇x log pdata(x) to sample from pdata(x), the samples obtained will not depend on π.

-> Langevin dynamics은 샘플링하는데 사용한다.

### 4 Noise Conditional Score Networks: learning and inference
We observe that perturbing data with random Gaussian noise makes the data distribution more amenable to score-based generative modeling. First, since the support of our Gaussian noise distribution is the whole space, the perturbed data will not be confined to a low dimensional manifold, which obviates difficulties from the manifold hypothesis and makes score estimation well-defined. Second, large Gaussian noise has the effect of filling low density regions in the original unperturbed data distribution; therefore score matching may get more training signal to improve score estimation. Furthermore, by using multiple noise levels we can obtain a sequence of noise-perturbed distributions that converge to the true data distribution. We can improve the mixing rate of Langevin dynamics on multimodal distributions by leveraging these intermediate distributions in the spirit of simulated annealing [30] and annealed importance sampling [37].

-> Gaussian noise distribution는 the whole space이고 가우시안 노이즈로 인해 perturbed된 data는 low dimensional manifold에 한정되지 않게 될 것이다. 그럼 score estimation에서 잘 정의할 수 있게 된다. 또한 original unperturbed data distribution(아직 노이즈가 추가되지 않은 본래의 데이터 분포)의 low density region을 큰 가우시안 노이즈로 채움으로써 score matching은 score estimation을 향상시키기 위한 training signal을 더 가질 수 있다. 

[Sliced Score Matching: A Scalable Approach to Density and Score Estimation](https://arxiv.org/pdf/1905.07088.pdf)

#### Implicit distributions have a tractable sampling process but without a tractable density. EX) GAN objective function

Besides parameter estimation in unnormalized models, score matching can also be used to estimate scores of implicit distributions, which are distributions that have a tractable sampling process but without a tractable density. For example, the distribution of random samples from the generator of a GAN (Goodfellow et al., 2014) is an implicit distribution. Implicit distributions can arise in many more situations such as the marginal distribution of a non-conjugate model (Sun et al., 2019), and models defined by complex simulation processes (Tran et al., 2017). In many cases learning and inference become intractable due to the need of optimizing an objective that involves the intractable density.


[Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.csail.mit.edu/papers/volume6/hyvarinen05a/hyvarinen05a.pdf)

#### 4.3 Conclusion (Statistical models, score matching, MCMC, computationally efficient

We have proposed a new method, score matching, to estimate statistical models in the case where the normalization constant is unknown. Although the estimation of the score function is computationally difficult, we showed that the distance of data and model score functions is very easy to compute. The main assumptions in the method are: 
1) all the variables are continuous-valued and defined over Rn, 
2) 
3) 2) the model pdf is smooth enough. Score matching provides a computationally simple yet locally consistent alternative to existing methods, such as MCMC and various approximative methods.

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

### 정상성과 비정상성 (Stationary & Non Stationary)

1. 정상/비정상성 과정 이란?

  ㅇ 정상성 (Stationary)
  
     - 통계적 성질이 시간에 따라 변하지 않음
     
     - 여러 시간 구간 마다 모두 동일한 통계적 특성을 갖음
     
     - 모든 시간에서 똑같은 성질을 갖는 랜덤변수로 관측됨

     * 정상성의 例)  동일 환경에서 되풀이되는 주사위 던지기 등

     * 한편, 확률적 의미 없이, 시간에 따른 규칙적인 거동은, ☞ 정상 상태 (Steady State) 참조

  ㅇ 비 정상성 (Non Stationary)
  
     - 정상성(Stationary)이 아니면, 비 정상성(Non Stationary)이라고 함
     
        . 즉, 시간에 따라 통계적 성질도 변해감

     * 비 정상성의 例)  예측이 어려운(변덕스러운) 날씨, 기후 등


2. 물리적 의미 (근사적으로)

  ㅇ 정상성(Stationary)은 안정적 물리적 현상 
  
     - 통계적 성질이 시간에 따라 변하지 않는 고정적인 특성

  ㅇ 비정상성(Nonstationary)은 불안정한 물리 현상
  
     - 통계적 성질이 시간에 따라 커지는 등 시변적인 특성

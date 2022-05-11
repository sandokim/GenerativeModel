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

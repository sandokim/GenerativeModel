# SOTA

[Projected GANs Converge Faster](https://arxiv.org/pdf/2111.01007.pdf)

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/projected GANs.jpg" width="50%">

[[75] D. Sungatullina, E. Zakharov, D. Ulyanov, and V. Lempitsky. Image manipulation with perceptual discriminators. In Proc. of the European Conf. on Computer Vision (ECCV), 2018.](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Diana_Sungatullina_Image_Manipulation_with_ECCV_2018_paper.pdf)

# Representation of Probability Distribution

### Explicit models: represent a probability density/mass function

gaussian distribution에서 invertible function들을 반복시켜 학습시키는 모델, jacobian가 잘 정의가 되어 있어 정확하게 explict하게 모델링한다.

Explicit model의 단점은 analytic한 function pθ(x)를 정확하게 수식으로 정의할 수 있어야한다. Explicit한 density function으로 근사한다. 모든 x에 대해서 integral을 계산해줘야하므로 Zθ가 intractable하다.

* Bayesian networks (e.g., VAEs)
* MRF
* Autoregressive models
* Flow models

### Implicit models: directly represent the sampling process

결과적으로 Sample을 뽑을 뿐, 직접적으로 modeling해주지는 않는다.

* GANs

### Score matching -> 어떤 모델링을 통해서 gardient logpθ(x)를 근사할 수 있으면 logpθ(x)를 알 수 있다. -> pθ(x)를 구할 수 있다.

당연한 의문점..? p(x)를 모르는데 gradient를 어떻게 구해서 매칭하나? Sampling을 어떻게 하지? --> Langevin Dynamics

--> 점을 무작위로 뿌린다음 Gradient의 field에 따라 어떻게 모이는지를 보고 그 점들이 가지는 분포를 보고 p(x)를 알 수 있겠다. Gradient Ascent하는 방법으로 막 뿌려놓고 그 점들이 어떻게 업데이트되는지 보는게 Langevin Dynamics의 특수 케이스다.

--> 문제는 아무런 perturbation이 없으면 항상 거의 deterministic하게 항상 정해져있는 Local한 maxima(=Maximum Liklikhood)에 가버린다.

--> 그래서 해결방안으로 Langevin dynmaics에서 노이즈에서 샘플링해가지고 추가로 넣어준다. Perturbation이 들어가면 noisy score를 따라가게 되면서 원래 데이터 분포를 좀 더 잘 근사한다.

#### Score Estimation

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Score Matching.jpg" width="50%">

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Score Matching 수식증명.jpg" width="50%">



[Read-through: Wasserstein GAN](https://www.alexirpan.com/2017/02/22/wasserstein-gan.html)

#### Wasserstein distance = kantorovich-Rubinstei = Optimal transport = Earth mover's distance 

[Introduction to the Wasserstein distance](https://www.youtube.com/watch?v=CDiol4LG2Ao)

#### High Dimension VS Low Dimension (Intersection X)

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/high_dimension_intersection.JPG" width="100%">

#### Wasserstein distance continuity and differentiability in loss function is crucial for learning

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Wasserstein_distance_continuity_and_differentiability.JPG" width="100%">

#### Supremum 예시, w는 weights이며 W는 모든 possible weights set (subspace, 하위집합)

Vector space K의 Subspace를 W라 정의하였다.

Subspace W에 속하는 모든 w에 대한 Maximum Expected Value는 f가 정의된 Vector space K의 supremum 상한(=최소상계, Least Upper Bound, LUB)보다는 항상 작거나 같아야한다.

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/supremum_ex.JPG" width="100%">

#### How Wasserstein Distance can be computed

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Wasserstein_model_train.JPG" width="100%">

#### Wasserstein Distance based model's convergence procedure

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Wasserstein_model_converge.JPG" width="100%">

#### Wasserstein Distance is differentiable nearly everywhere compared to GANs which use JS divergence (JS divergence can make a gradient 0..!)

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Wasserstein_distance_is_differentiable.JPG" width="100%">

#### WGAN VS GAN 

WGAN gives a reasonably nice gradient over everything, whereas GAN discriminator does so in a way that makes gradients vanish over most of the space(=mode collapse). 

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/WGAN_VS_GAN.JPG" width="50%">

#### Wasserstein Distance Discrete

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Wasserstein_distance_discrete.JPG" width="40%" align='left'/>

#### Wasserstein Distance Continuous

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/Wasserstein_distance_continuous.JPG" width="50%" align='center'/>

#### Joint distribution Gamma

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/joint_distribution_gamma.JPG" width="80%">

#### Earth Mover distance, Lipschitz condition 1

[Wasserstein GANs with Gradient Penalty](https://www.youtube.com/watch?v=v6y5qQ0pcg4)

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/EM.jpg" width="80%">

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/W-loss,bce-loss.jpg" width="80%">

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/1-L continuous.jpg" width="80%">

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/W-Loss.jpg" width="80%">

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/summary1.jpg" width="80%">

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/summary2.jpg" width="80%">


[f-GAN: Training Generative Neural Samplers using Variational Divergence Minimization](https://arxiv.org/pdf/1606.00709.pdf)

반연속, lower-semicontinuous function

[Semi-continuity, 위에서 반연속, 아래서 반연속](https://ko.wikipedia.org/wiki/%EB%B0%98%EC%97%B0%EC%86%8D_%ED%95%A8%EC%88%98)

### 위에서 반연속, upper semicontinuous

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/위에서 반연속.JPG" width="30%" align='left'/>

### 아래서 반연속, lower semicontinuous

<img src="https://github.com/hyeseongkim0/Generative-Model/blob/main/images/아래서 반연속.JPG" width="30%" align='center'/>

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
     

# Reference

#### [Generative Adversarial Perturbations](https://openaccess.thecvf.com/content_cvpr_2018/papers/Poursaeed_Generative_Adversarial_Perturbations_CVPR_2018_paper.pdf)

[NIPS 2017: Non-targeted Adversarial Attack](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack)

* Non-targeted Adversarial Attack. The goal of the non-targeted attack is to slightly modify source image in a way that image will be classified incorrectly by generally unknown machine learning classifier.
* Targeted Adversarial Attack. The goal of the targeted attack is to slightly modify source image in a way that image will be classified as specified target class by generally unknown machine learning classifier.
* Defense Against Adversarial Attack. The goal of the defense is to build machine learning classifier which is robust to adversarial example, i.e. can classify adversarial images correctly.

References

[1] N. Akhtar, J. Liu, and A. Mian. Defense against universal
adversarial perturbations. arXiv preprint arXiv:1711.05929,
2017. 2

[2] A. Arnab, O. Miksik, and P. H. Torr. On the robustness of
semantic segmentation models to adversarial attacks. arXiv
preprint arXiv:1711.09856, 2017. 2

[3] A. Athalye, N. Carlini, and D. Wagner. Obfuscated gradients
give a false sense of security: Circumventing defenses to adversarial examples. arXiv preprint arXiv:1802.00420, 2018.
2

[4] A. Athalye and I. Sutskever. Synthesizing robust adversarial
examples. arXiv preprint arXiv:1707.07397, 2017. 2

[5] V. Badrinarayanan, A. Kendall, and R. Cipolla. Segnet: A
deep convolutional encoder-decoder architecture for image
segmentation. arXiv preprint arXiv:1511.00561, 2015. 1

[6] S. Baluja and I. Fischer. Adversarial transformation networks: Learning to generate adversarial examples. arXiv
preprint arXiv:1703.09387, 2017. 2

[7] A. N. Bhagoji, W. He, B. Li, and D. Song. Exploring the
space of black-box attacks on deep neural networks. arXiv
preprint arXiv:1712.09491, 2017. 3, 5, 7

[8] N. Carlini and D. Wagner. Towards evaluating the robustness
of neural networks. In Security and Privacy (SP), 2017 IEEE
Symposium on, pages 39–57. IEEE, 2017. 2, 3, 5, 6

[9] L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and
A. L. Yuille. Deeplab: Semantic image segmentation with
deep convolutional nets, atrous convolution, and fully connected crfs. arXiv preprint arXiv:1606.00915, 2016. 1

[10] P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, and C.-J. Hsieh. Zoo:
Zeroth order optimization based black-box attacks to deep
neural networks without training substitute models. In Proceedings of the 10th ACM Workshop on Artificial Intelligence
and Security, pages 15–26. ACM, 2017. 7

[11] M. Cisse, Y. Adi, N. Neverova, and J. Keshet. Houdini:
Fooling deep structured prediction models. arXiv preprint
arXiv:1707.05373, 2017. 2

[12] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler,
R. Benenson, U. Franke, S. Roth, and B. Schiele. The
cityscapes dataset for semantic urban scene understanding.
In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 3213–3223, 2016. 7

[13] N. Das, M. Shanbhogue, S.-T. Chen, F. Hohman, S. Li,
L. Chen, M. E. Kounavis, and D. H. Chau. Shield: Fast, practical defense and vaccination for deep learning using jpeg
compression. arXiv preprint arXiv:1802.06816, 2018. 2

[14] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. FeiFei. Imagenet: A large-scale hierarchical image database.
In Computer Vision and Pattern Recognition, 2009. CVPR
2009. IEEE Conference on, pages 248–255. IEEE, 2009. 5

[15] E. L. Denton, S. Chintala, R. Fergus, et al. Deep generative image models using a laplacian pyramid of adversarial
networks. In Advances in neural information processing systems, pages 1486–1494, 2015. 3

[16] G. S. Dhillon, K. Azizzadenesheli, Z. C. Lipton, J. Bernstein,
J. Kossaifi, A. Khanna, and A. Anandkumar. Stochastic activation pruning for robust adversarial defense. arXiv preprint
arXiv:1803.01442, 2018. 2

[17] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu,
D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In Advances in neural information
processing systems, pages 2672–2680, 2014. 3

[18] I. J. Goodfellow, J. Shlens, and C. Szegedy. Explaining and harnessing adversarial examples. arXiv preprint
arXiv:1412.6572, 2014. 2, 5, 7

[19] C. Guo, M. Rana, M. Cisse, and L. van der Maaten. Coun- ´
tering adversarial images using input transformations. arXiv
preprint arXiv:1711.00117, 2017. 2

[20] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
770–778, 2016. 1

[21] X. Huang, Y. Li, O. Poursaeed, J. Hopcroft, and S. Belongie.
Stacked generative adversarial networks. arXiv preprint
arXiv:1612.04357, 2016. 3

[22] P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Imageto-image translation with conditional adversarial networks.
arXiv preprint arXiv:1611.07004, 2016. 3

[23] J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for
real-time style transfer and super-resolution. In European
Conference on Computer Vision, pages 694–711. Springer,
2016. 3

[24] D. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. 2

[25] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet
classification with deep convolutional neural networks. In
Advances in neural information processing systems, pages
1097–1105, 2012. 1

[26] A. Kurakin, I. Goodfellow, and S. Bengio. Adversarial examples in the physical world. arXiv preprint arXiv:1607.02533,
2016. 2, 3

[27] A. Kurakin, I. Goodfellow, and S. Bengio. Adversarial machine learning at scale. arXiv preprint arXiv:1611.01236,
2016. 2, 3

[28] A. B. L. Larsen, S. K. Sønderby, H. Larochelle, and
O. Winther. Autoencoding beyond pixels using a learned
similarity metric. arXiv preprint arXiv:1512.09300, 2015. 3

[29] Y. Liu, X. Chen, C. Liu, and D. Song. Delving into transferable adversarial examples and black-box attacks. arXiv
preprint arXiv:1611.02770, 2016. 7

[30] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional
networks for semantic segmentation. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition, pages 3431–3440, 2015. 1, 7

[31] J. Lu, H. Sibai, E. Fabry, and D. Forsyth. No need to
worry about adversarial examples in object detection in autonomous vehicles. arXiv preprint arXiv:1707.03501, 2017.
2

[32] X. Ma, B. Li, Y. Wang, S. M. Erfani, S. Wijewickrema, M. E.
Houle, G. Schoenebeck, D. Song, and J. Bailey. Characterizing adversarial subspaces using local intrinsic dimensionality. arXiv preprint arXiv:1801.02613, 2018. 2
4430

[33] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and
A. Vladu. Towards deep learning models resistant to adversarial attacks. arXiv preprint arXiv:1706.06083, 2017. 2

[34] J. H. Metzen, M. C. Kumar, T. Brox, and V. Fischer. Universal adversarial perturbations against semantic image segmentation. arXiv preprint arXiv:1704.05712, 2017. 2, 7, 8

[35] S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, and
P. Frossard. Universal adversarial perturbations. arXiv
preprint arXiv:1610.08401, 2016. 1, 2, 5

[36] S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, P. Frossard,
and S. Soatto. Analysis of universal adversarial perturbations. arXiv preprint arXiv:1705.09554, 2017. 2

[37] S.-M. Moosavi-Dezfooli, A. Fawzi, and P. Frossard. Deepfool: a simple and accurate method to fool deep neural networks. In Proceedings of the IEEE Conference on Computer
Vision and Pattern Recognition, pages 2574–2582, 2016. 2

[38] K. R. Mopuri, U. Garg, and R. V. Babu. Fast feature fool: A
data independent approach to universal adversarial perturbations. arXiv preprint arXiv:1707.05572, 2017. 2

[39] A. Nguyen, J. Yosinski, and J. Clune. Deep neural networks
are easily fooled: High confidence predictions for unrecognizable images. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition, pages 427–436,
2015. 2

[40] N. Papernot, P. McDaniel, and I. Goodfellow. Transferability
in machine learning: from phenomena to black-box attacks
using adversarial samples. arXiv preprint arXiv:1605.07277,
2016. 7

[41] N. Papernot, P. McDaniel, I. Goodfellow, S. Jha, Z. B. Celik, and A. Swami. Practical black-box attacks against deep
learning systems using adversarial examples. arXiv preprint
arXiv:1602.02697, 2016. 7

[42] A. Prakash, N. Moran, S. Garber, A. DiLillo, and J. Storer.
Deflecting adversarial attacks with pixel deflection. arXiv
preprint arXiv:1801.08926, 2018. 2

[43] A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434, 2015. 3

[44] A. Raghunathan, J. Steinhardt, and P. Liang. Certified
defenses against adversarial examples. arXiv preprint
arXiv:1801.09344, 2018. 2

[45] A. S. Rakin, Z. He, B. Gong, and D. Fan. Robust preprocessing: A robust defense method against adversary attack. arXiv preprint arXiv:1802.01549, 2018. 2

[46] O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and
Computer-Assisted Intervention, pages 234–241. Springer,
2015. 3

[47] A. Roy, C. Raffel, I. Goodfellow, and J. Buckman. Thermometer encoding: One hot way to resist adversarial examples. 2018. 2

[48] P. Samangouei, M. Kabkab, and R. Chellappa. Defense-gan:
Protecting classifiers against adversarial attacks using generative models. 2018. 2

[49] K. Simonyan and A. Zisserman. Very deep convolutional
networks for large-scale image recognition. arXiv preprint
arXiv:1409.1556, 2014. 1

[50] Y. Song, T. Kim, S. Nowozin, S. Ermon, and N. Kushman.
Pixeldefend: Leveraging generative models to understand
and defend against adversarial examples. arXiv preprint
arXiv:1710.10766, 2017. 2

[51] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed,
D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich.
Going deeper with convolutions. In Proceedings of the
IEEE conference on computer vision and pattern recognition, pages 1–9, 2015. 1

[52] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna.
Rethinking the inception architecture for computer vision.
In Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition, pages 2818–2826, 2016. 1

[53] C. Szegedy, W. Zaremba, I. Sutskever, J. Bruna, D. Erhan,
I. Goodfellow, and R. Fergus. Intriguing properties of neural
networks. arXiv preprint arXiv:1312.6199, 2013. 1, 2, 7

[54] F. Tramer, A. Kurakin, N. Papernot, D. Boneh, and P. Mc- `
Daniel. Ensemble adversarial training: Attacks and defenses.
arXiv preprint arXiv:1705.07204, 2017. 2

[55] D. Vijaykeerthy, A. Suri, S. Mehta, and P. Kumaraguru.
Hardening deep neural networks via adversarial model cascades. arXiv preprint arXiv:1802.01448, 2018. 2

[56] T.-W. Weng, H. Zhang, P.-Y. Chen, J. Yi, D. Su, Y. Gao, C.-
J. Hsieh, and L. Daniel. Evaluating the robustness of neural
networks: An extreme value theory approach. arXiv preprint
arXiv:1801.10578, 2018. 2

[57] C. Xie, J. Wang, Z. Zhang, Z. Ren, and A. Yuille. Mitigating adversarial effects through randomization. arXiv preprint
arXiv:1711.01991, 2017. 2

[58] C. Xie, J. Wang, Z. Zhang, Y. Zhou, L. Xie, and A. Yuille.
Adversarial examples for semantic segmentation and object
detection. arXiv preprint arXiv:1703.08603, 2017. 2, 7

[59] F. Yu and V. Koltun. Multi-scale context aggregation by dilated convolutions. arXiv preprint arXiv:1511.07122, 2015.
1

[60] H. Zhao, J. Shi, X. Qi, X. Wang, and J. Jia. Pyramid scene
parsing network. arXiv preprint arXiv:1612.01105, 2016. 1

[61] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired imageto-image translation using cycle-consistent adversarial networks. arXiv preprint arXiv:1703.10593, 2017. 3




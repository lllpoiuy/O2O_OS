Main components:
+ critic_loss yes
+ actor_loss yes
+ sample_action
+ sample_batch
+ create_networks
+ imitation controller
+ offline controller
+ warm-up controller
+ online controller
+ concepts (dormant ratio, simplicity bias)
+ dataset & env


Plugin-compenents:
+ expectile regression (critic_loss)
+ conservatism term (critic_loss)
+ calibrated term (critic_loss)
+ auxiliary one-step denoiser (sample_action, actor_loss, create_networks)
+ action chunking
+ best-of-n sampling (sample_action)
+ action gradient (sample_action)
+ base policy bootstraped exploration (sample_action)
+ noise guided exploration (sample_action, create_networks)
+ auxiliary normal learner (actor_loss, sample_action, create_networks)
+ state-action value offset function (create_networks) (see Fisher-BRC)
+ dual policy optimistic exploration (create_networks) (see BRO)
+ pessimistic/non-pessimistic quantile Q-value (create_networks) (see BRO)



Networks:
+ Q-ensemble
+ layernorm critic
+ normal actor
+ flow matching actor
+ diffusion actor
+ vla actor
+ MoE
+ LERP/Residual Feedforward Block/linear+scalar(see SimBa2)

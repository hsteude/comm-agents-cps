from comm_agents.lightning_module import LitModule
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from examples.three_tank.data_module import ThreeTankDataModule
import torch

seed_everything(42)


def train():
    hparams = dict(
        enc_input_size=3,
        enc_hidden_size=20,
        enc_rnn_layers=3,
        latent_dim=5,
        filt_initial_log_var=-10,
        filt_num_decoders=4,
        dec_num_question_inputs=0,
        dec_hidden_size=10,
        dec_num_hidden_layers=3,
        dec_out_dim=1,
        batch_size=12,
        dl_num_workers=2,
        validation_split=0.05,
        learning_rate=1e-3,
        beta=0.001
    )
    # debugging forward pass
    lit_module = LitModule(**hparams)

    gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=10_001
        # callbacks=[
            # # BetaIncreaseCallBack(initial_beta=0, beta_max=1,
                            # # number_steps=20, increase_after_n_epochs=200)
            # ]
    )
    dm = ThreeTankDataModule(**hparams)
    trainer.fit(lit_module, dm)

if __name__ == '__main__':
    train()

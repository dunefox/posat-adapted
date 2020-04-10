function erw()
  for i in 1:2
      for weight=0.0:0.01:0.08
          open("finalresults/finetune_erw$(i)_train_$(weight).sh", "w") do file
              write(file, """#!/bin/sh
              #SBATCH --gres=gpu:0
              #SBATCH --partition=Luna,Gobi
              #SBATCH --output=results/erw$(i)_train_$(weight).out
              #SBATCH --job-name="erw$(i)_train_$(weight)"

              /home/f/fuchsp/.pyenv/shims/python ../model/train_finetune.py --data_dir /big/f/fuchsp/data/datasets/train_tr --vocab_dir ../model/dataset/vocab --model_dir ./saved_models/clean_ds_train --erw$(i) True --weight $(weight) --id erw$(i)_train_$(weight) --info "Position-aware attention model"

              """)
          end
      end
  end
end

function bias_fix()
  for weight=0.0:0.5:3.0
      open("finalresults/bias_fix_train_$(weight).sh", "w") do file
          write(file, """#!/bin/sh
          #SBATCH --gres=gpu:0
          #SBATCH --partition=Luna,Gobi
          #SBATCH --output=results/bias_fix_train_$(weight)
          #SBATCH --job-name="bias_fix_train_$(weight)"

          /home/f/fuchsp/.pyenv/shims/python ../model/train_bias.py --data_dir /big/f/fuchsp/data/datasets/train_ds --vocab_dir ../model/dataset/vocab --bias_weight $(weight) --id bias_fix_train_$(weight) --info "Position-aware attention model"

          """)
      end
  end
end

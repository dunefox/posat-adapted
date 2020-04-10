function bias_set()
  for train_weight=0.0:0.5:3.0, test_weight=0.0:0.5:3.0
      open("results/bias_set_test_train$(train_weight)_test$(test_weight).sh", "w") do file
          write(file, """#!/bin/sh
          #SBATCH --gres=gpu:0
          #SBATCH --partition=Luna,Gobi
          #SBATCH --output=results/bias_set_test_train$(train_weight)_test$(test_weight).out
          #SBATCH --job-name="bias_set_test_train$(train_weight)_test$(test_weight)"

          python ../model/eval_bias.py saved_models/clean_ds_train --data_dir /big/f/fuchsp/data/datasets/train_tr --dataset test --bias_mode set --bias_set_train $(train_weight) --bias_fix_test -1.0 --bias_set_test $(test_weight) --model_desc bias_set_test_$(train_weight)_$(test_weight)
          """)
      end
  end
end

function bias_fix()
  for train_weight=0.0:0.5:3.0, test_weight=0.0:0.5:3.0
      open("results/bias_fix_test_train$(train_weight)_test$(test_weight).sh", "w") do file
          write(file, """#!/bin/sh
          #SBATCH --gres=gpu:0
          #SBATCH --partition=Luna,Gobi
          #SBATCH --output=results/bias_fix_test_train$(train_weight)_test$(test_weight).out
          #SBATCH --job-name="bias_fix_test_train$(train_weight)_test$(test_weight)"

          python ../model/eval_bias.py saved_models/bias_fix_train_$(train_weight) --data_dir /big/f/fuchsp/data/datasets/train_tr --dataset test --bias_mode fix --bias_fix_test $(test_weight) --bias_set_test -1.0 --bias_set_train -1.0 --model_desc bias_fix_test_$(train_weight)_$(test_weight)
          """)
      end
  end
end

function erw2()
  for train_weight=0.0:0.01:0.08
      open("results/erw2_test_$(train_weight).sh", "w") do file
          write(file, """#!/bin/sh
          #SBATCH --gres=gpu:0
          #SBATCH --partition=Luna,Gobi
          #SBATCH --output=results/erw2_test_$(train_weight).out
          #SBATCH --job-name="erw2_test_$(train_weight)"

          python ../model/eval.py saved_models/erw2_train_$(train_weight) --data_dir /big/f/fuchsp/data/datasets/train_tr --eval_mode erw2 --dataset test --model_desc erw2_test_$(train_weight)

          """)
      end
  end
end

function erw1()
  for train_weight=0.0:0.01:0.08
      open("results/erw1_test_$(train_weight).sh", "w") do file
          write(file, """#!/bin/sh
          #SBATCH --gres=gpu:0
          #SBATCH --partition=Luna,Gobi
          #SBATCH --output=results/erw1_test_$(train_weight).out
          #SBATCH --job-name="erw1_test_$(train_weight)"

          python ../model/eval.py saved_models/erw1_train_$(train_weight) --data_dir /big/f/fuchsp/data/datasets/train_tr --eval_mode erw1 --dataset test --model_desc erw1_test_$(train_weight)

          """)
      end
  end
end


open("results/finetune_attn_test.sh", "w") do file
  write(file, """#!/bin/sh
    #SBATCH --gres=gpu:0
    #SBATCH --partition=Luna,Gobi
    #SBATCH --output=results/finetune_attn_test
    #SBATCH --job-name="finetune_attn"

    python ../model/eval.py saved_models/finetune_attn_train --data_dir /big/f/fuchsp/data/datasets/train_tr --eval_mode standard --dataset test --model_desc finetune_attn_test

    """)
end

open("results/joint_training_test.sh", "w") do file
  write(file, """#!/bin/sh
    #SBATCH --gres=gpu:0
    #SBATCH --partition=Luna,Gobi
    #SBATCH --output=results/joint_training_test
    #SBATCH --job-name="joint_training_test"

    python ../model/eval.py saved_models/joint_training_train --data_dir /big/f/fuchsp/data/datasets/train_tr --eval_mode standard --dataset test --model_desc joint_training_test

    """)
end


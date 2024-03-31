

import pandas as pd
import logging

from sympy import false, true

from seq2seq_model import Seq2SeqModel
import argparse
if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Specify dataset location.')
    parser.add_argument('--dataset_loc', type=str, help='Location of the dataset file', required=True)
    parser.add_argument('--fold_num', type=str, help='train fold', required=True)

    # Parse the command line arguments
    args = parser.parse_args()

    # Use args.dataset_loc to access the specified dataset location
    dataset_location = args.dataset_loc  # This variable now holds the path to the dataset file
    fold_num = args.fold_num
    # The rest of your code, now use 'dataset_location' instead of hard-coded file paths
    # For example, instead of opening the dataset file with a hard-coded path:
    # f = open('./data/test_causal_timebank.csv', 'r',encoding='utf-8')
    # Use the specified dataset location:
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)


    data = pd.read_csv(f"{dataset_location}/train_fold_{fold_num}.csv", sep=',',encoding = 'utf-8',usecols=[0,1]).values.tolist()
    # data = pd.read_csv("./data/train_eventstoryline.csv", sep=',',encoding = 'UTF-8',usecols=[0,1]).values.tolist()
    df = pd.DataFrame(data, columns=["input_text", "target_text"])
    train_df=df[:int(df.shape[0] * 0.8)]
    eval_df=df[~df.index.isin(train_df.index)]


    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "max_seq_length": 160,
        "train_batch_size": 36,
        "num_train_epochs": 10,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_during_training": True,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "use_multiprocessing": False,
        "max_length": 32,
        "manual_seed": 6666,
        "save_steps": 11898,
        "gradient_accumulation_steps": 1,
        "best_model_dir":f"{dataset_location}/model_{fold_num}",
        # "output_dir": "./exp/eventstoryline_relpos_regu",
        "output_dir":"./exp/causal_timebank_relpos_regu",
    }

    # Initialize model
    model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name="facebook/bart-base",
        args=model_args,
        cuda_device = 0 ,
        regularization = True,
        # use_cuda=False,
    )



    # Train the model
    model.train_model(train_df, eval_data=eval_df)

    # Evaluate the model
    results = model.eval_model(eval_df)

    # Use the model for prediction

    print(model.predict(["During Lohan's stay at the famed Betty Ford Center in 2010, Lohan was accused of attacking a female staffer"]))



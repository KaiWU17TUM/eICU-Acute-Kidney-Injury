from train_SVM_raw_data import *

if __name__ == '__main__':
    np.random.seed(0)
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--pred",
        nargs="*",
        type=int,
        default=[0, 6, 12],
    )
    CLI.add_argument(
        "--imp",
        nargs="*",
        type=str,
        default=["mean", "zero"],
    )
    CLI.add_argument(
        "--rr",
        nargs="*",
        type=int,
        default=[5, 30, 60]
    )
    args = CLI.parse_args()

    pred_list = args.pred
    imp_list = args.imp
    rr_list = args.rr

    for pred in pred_list:
        for imp in imp_list:
            for resample_rate in rr_list:
                print(f'##################### PRED {pred} - IMP {imp} - SAMPLERATE {resample_rate} #####################')

                root_folder = os.path.join(os.path.abspath('.'), f'processed/patient_data/dx_pred_{pred}_12_3src')
                data = np.load(os.path.join(root_folder, f'ts_aligned_stacked_{pred}_{imp}_{resample_rate}.npz'))
                X_test = data['X_test']
                y_test = data['y_test']
                del data

                save_model_path = 'models/SVM/SVM_raw/'
                save_model_path = save_model_path + f'pred{pred}-imp{imp}-sampling{resample_rate}-raw.sav'
                clf = pickle.load(open(save_model_path, 'rb'))

                y_pred = clf.predict(X_test)
                METRICS = {
                    'acc': accuracy_score,
                    'recall': recall_score,
                    'precision': precision_score,
                    'f1': f1_score,
                    'auroc': roc_auc_score
                }

                save_path = os.path.join('models/SVM/', 'model_performance_raw')
                Path(save_path).mkdir(parents=True, exist_ok=True)

                columns = ['test_roc', 'test_recall', 'test_precision', 'test_f1', 'test_acc', 'test_loss']

                res = []
                for metric in METRICS:
                    metric_val = METRICS[metric](y_test, y_pred)
                    res.append(metric_val)
                    print(f'{metric}: {metric_val}')
                res.append(np.nan)
                res_dict = {(pred, imp, resample_rate): res}
                df = pd.DataFrame.from_dict(res_dict, orient='index', columns=columns)
                df.to_csv(os.path.join(save_path, f'{pred}-{imp}-{resample_rate}.csv'), sep=';')
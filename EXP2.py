from neural_fields.hypernet_studies.sample_exp_two_three import sample_Exp2, sample_Exp3
import time


if __name__ == '__main__':

    start_time = time.time()
    sample_Exp2()
    # sample_Exp3()

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Total execution time: {elapsed_time:.4f} seconds")


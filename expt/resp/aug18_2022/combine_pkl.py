import pickle
from typing import Optional

def combine_pickle_files(pklfname_main: str, pklfname_last: str, pklfname_circ10: Optional[str] = None) -> None:
    with open(pklfname_main + ".pkl", "rb") as f:
        pkl_data_main = pickle.load(f)
    with open(pklfname_last + ".pkl", "rb") as f:
        pkl_data_last = pickle.load(f)
    if pklfname_circ10 is not None:
        with open(pklfname_circ10 + ".pkl", "rb") as f:
            pkl_data_circ10 = pickle.load(f)

    pkl_data_last["labels"] = [s.replace("circ0u1d", "circlast") for s in pkl_data_last["labels"]]
    pkl_data_main["labels"] += pkl_data_last["labels"]
    pkl_data_main["results"] += pkl_data_last["results"]
    for i, (label, result) in enumerate(zip(pkl_data_main["labels"], pkl_data_main["results"])):
        if len(list(result.keys())[0]) == 3:
            print("label = ", label)
            result_new = dict()
            for key, value in result.items():
                result_new['0' + key] = value
            pkl_data_main["results"][i] = result_new
        if pklfname_circ10 is not None:
            if label[:7] == "circ10/":
                index = pkl_data_circ10["labels"].index(label)
                print(index, label)
                pkl_data_main["results"][i] = pkl_data_circ10["results"][index]
        

    with open(pklfname_last + "_comb.pkl", "wb") as f:
        pickle.dump(pkl_data_main, f)

def main() -> None:
    combine_pickle_files("../aug13_2022_traj/nah_by_depth_0813", "nah_resp_alltomorun")
    combine_pickle_files("../aug13_2022_traj/nah_2q_by_depth_0813", "nah_resp_alltomo2qrun", "nah_resp_circ0u1d2qrun")
    combine_pickle_files("../aug13_2022_traj/kh_by_depth_0813", "kh_resp_alltomorun")
    combine_pickle_files("../aug13_2022_traj/kh_2q_by_depth_0813", "kh_resp_alltomo2qrun")

if __name__ == "__main__":
    main()
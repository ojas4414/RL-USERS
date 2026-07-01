import pandas as pd
import json


def load_reviews(file:str)->pd.DataFrame:
    rows=[]
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            p=json.loads(line)
            rows.append({
                "user_id":p.get("user_id"),
                "item_id": p.get("parent_asin"),
                "rating": p.get("rating"),
                "timestamp": p.get("timestamp")

            })

    return pd.DataFrame(rows)



def filter_(d:pd.DataFrame)->pd.DataFrame:
    user_counts=d["user_id"].value_counts()
    item_counts=d["item_id"].value_counts()

    valid_users=user_counts[user_counts>=5].index # isnide []  we have boolean as answer this return only valid user id and count .index then strips values and return id 
    valid_items=item_counts[item_counts>=5].index
    # mask = boolean Series, same index as df, one True/False per row
    # df[mask] checks each row's index label against the mask —
    # keeps the row if True, drops it if False

    filtered=d [
        d["user_id"].isin(valid_users)&
        d["item_id"].isin(valid_items)
    ]
    return filtered

def sort_split(d:pd.DataFrame):
    d=d.sort_values("timestamp").reset_index(drop=True)

    train_cutoff=d["timestamp"].quantile(0.80)
    val_cutoff = d["timestamp"].quantile(0.90)
    return d,train_cutoff,val_cutoff

def build(d:pd.DataFrame,window_size: int=3):
    pairs=[]
    for user_id,group in d.groupby("user_id"):
        items= group["item_id"].tolist()
        timestamps=group["timestamp"].tolist()

        if len(items)< window_size +1:
            continue

        for i in range(len(items)-window_size):#why this range as every window needs winddo size + 1 extra for the action 
            state = items[i:i+ window_size]
            action =items[i + window_size]
            action_ts = timestamps[i+ window_size]
            pairs.append({
                "user_id": user_id,
                "state": state,
                "action": action,
                "timestamp": action_ts

            })
    return pairs

def dataset(pairs:list,train_cutoff, val_cutoff):
    train,val,test=[],[],[]

    for i in pairs:
        if i["timestamp"]<=train_cutoff:
            train.append(i)
        elif i["timestamp"]<=val_cutoff:
            val.append(i)
        else:
            test.append(i)

    return train,val,test






    
    


    









     


# implement a train bpe (Byte Pair Encoding )
# get vocab,merge tokens
import regex as re 
from pathlib import Path
from collections import Counter
import pickle
import cProfile
from memory_profiler import memory_usage
# import pstats




## pretokenization function: 

def pretokenization(input_path:str 
                    ,special_token: list[str]|None=None):
    Pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    text=Path(input_path).read_text(encoding='utf-8')

    if special_token:
        motif="|".join(
            re.escape(token) for token in sorted(special_token,key=len, reverse=True)
            )
        chunks=re.split(motif,text)
    else: 
        chunks=[text]

    Pattern=re.compile(Pat)
    counts=Counter()
    for chunk in chunks:
        for m in Pattern.finditer(chunk):
            tokens=m.group()
            counts[tokens]+=1 # init a empty dict which init counter =0  ,  key : value = computer + incrementor (if key 've find)
    return counts

# convert counts  (type (str)-< text)

def convert_to_bytes(counts):
    bytes_counts=Counter()
    for data, frequency in counts.items():
        data=data.encode(encoding='utf-8')
        a= tuple(bytes([b]) for b in data)
        bytes_counts[a]+=frequency 
    return bytes_counts

# counts les pairs

def counts_pairs (bytes_counts):
    pairs_counts= Counter()

    for sequences, frequency in bytes_counts.items():
        for a,b in zip(sequences,sequences[1:]):
            # print((a,b))
            pairs_counts[(a,b)]+=frequency 
    return pairs_counts


def merge_pair(sequence,pairs):
    result=[]
    i=0
    while i<len(sequence):
        if (
            i+1<len(sequence)
            and (sequence[i],sequence[i+1])==pairs
        ):
            result.append(sequence[i]+sequence[i+1])
            i+=2
        else : 
            result.append(sequence[i])
            i+=1
    return tuple(result)

# update sequence 
# update
def update_sequences(byte_counts,best_pairs):
    byte_news=Counter()
    for sequence,frequency in byte_counts.items():
        new_sequence=merge_pair(sequence,best_pairs)
        byte_news[new_sequence]+=frequency
    return byte_news


def save_vocabulary(vocab:dict[int,bytes]):

    vocab_dir=Path('vocab_dir')
    vocab_dir.mkdir(exist_ok=True)
    
    with open(vocab_dir / "vocab_gle.pkl",'wb') as f :
        pickle.dump(vocab,f)

# cProfile find the bottleneck

def bottleneck(funct,*args,**kwargs): # *args = arguments non nommé, *kwargs=arguments nommés eg : age='23'
    Path("bottleneck").mkdir(exist_ok=True)

    profile=cProfile.Profile()
    profile.enable()
    mem,result=memory_usage(
        (funct,args,kwargs),
        retval=True,
        max_usage=True

    )
    
    profile.disable()
    profile.dump_stats("bottleneck/train_bpe.prof")
    print (f"Max Memory : {mem: .2f}MiB ")

    return result

    
    
    



def train_bpe(
        input_path: str,
        vocab_size: int,
        special_token:list[str]

):
    """
    return 
    vocab : dict[int,bytes] 
    merges: list[tuple[bytes,bytes]]
    """

    # Vocabulary initialization  

    vocab= {
        i: bytes([i]) for i in range (256)
    }
    special_token =list(dict.fromkeys(special_token))
    # add a special token in the INIT vocab
    for token in special_token : 
        vocab[len(vocab)]=token.encode('utf-8')

    merge=[]

    if vocab_size< len (vocab):
        raise ValueError ('Input vocabulary_size is  low') 

    counts=pretokenization(input_path,special_token) 
    # convert to byte 
    bytes_counts=convert_to_bytes(counts)

    while len(vocab)<vocab_size:
        
        pairs_counts=counts_pairs(bytes_counts)
        if not pairs_counts:
            break


        # best pairs 
        best_pairs= max(
            pairs_counts,
            key=lambda pair: (pairs_counts[pair],pair)
        ) # max parcours and  use key = to get a score and score use for selection by max
        new_tokens=best_pairs[0]+best_pairs[1]
        merge.append(best_pairs)

        vocab[len(vocab)]=new_tokens
        # actualiser 
        bytes_counts=update_sequences(
            bytes_counts,
            best_pairs
        )
    save_vocabulary(vocab=vocab)

    return vocab,merge

#another execute command 
# vocab,merge=train_bpe(
#     r"input path",
#     300,
#     special_token=["<|endoftext|>"]
# )

# print("merges learning :")
# print (vocab)

# bottleneck command 
# if __name__=="__main__":
#     bottleneck(train_bpe,input_path=r"",vocab_size=300,special_token=["<|endoftext|>"])






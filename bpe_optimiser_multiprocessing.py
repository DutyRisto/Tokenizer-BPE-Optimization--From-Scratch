# implementer a train bpe (Byte Pair Encoding ) 
# optimisation for multiprocessing

# get vocab,merge tokens
import regex as re 
from pathlib import Path
from collections import Counter
import pickle
import cProfile
from memory_profiler import memory_usage
import pstats
from multiprocessing import Pool
from pretokenization_example import find_chunk_boundaries
import os


        


## pretokenization function: 

def pretokenization_chunk(text:str 
                    ,special_token: list[str]|None=None):
    Pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    

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
# worker
def process_chunk (
        input_path:str,
        start:int,
        end:int,
        special_token: list[str]|None
)->Counter:

    with open(input_path,'rb') as f :
        f.seek(start)
        chunk=f.read(end-start)
        text=chunk.decode(encoding='utf-8')

    return pretokenization_chunk(
        text=text, 
        special_token=special_token
    )
def pretokenization(
        input_path,
        special_token,
        num_processes
        
)->Counter: 

    if num_processes is None : 
        num_processes = os.cpu_count() or 1

    split_token=b"<|endoftext|>"

    # find a trusty boundary
    with open(input_path,'rb') as f:
        boundaries=find_chunk_boundaries(
            f,
            desired_num_chunks=num_processes,
            split_special_token=split_token,
        )
    # jobs for each CPU 
    job=[]
    for start,end in zip(boundaries[:-1],boundaries[1:]):
        job.append(
            (input_path,start,end,special_token)
        )

    # execute each chunk in  parallele
    with Pool(processes=num_processes)as pool:
        result=pool.starmap(process_chunk,job)

    # merge the counter
    totals_counts=Counter()

    for counts in result : 
        totals_counts.update(counts) # similar to totals_counts[(counts)] +=1  

    return totals_counts
         

    
    
    
# convert counts  (type (str)-< texte)

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


def save_vocabulary(vocab:dict[int,bytes],merge):

    vocab_dir=Path('vocab_dir')
    vocab_dir.mkdir(exist_ok=True)
    merge_dir=Path('merge_dir')
    merge_dir.mkdir(exist_ok=True)
    
    with open(vocab_dir / "vocab_gle.pkl",'wb') as f :
        pickle.dump(vocab,f)
    with open(merge_dir / "merge_gle.pkl",'wb') as f :
        pickle.dump(merge,f)
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
        special_token:list[str],
        num_processes:int

):
    """
    return 
    vocab : dict[int,bytes] 
    merges: list[tuple[bytes,bytes]]
    """

    # initialisation du vocabulaire

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

    counts=pretokenization(input_path,special_token,num_processes) 
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
    save_vocabulary(vocab=vocab,merge=merge)

    return vocab,merge

# vocab,merge=train_bpe(
#     r"C:\Users\lenovo\Documents\Cours LLM Stanford\code_apprentissage\data\sample.text",
#     300,
#     special_token=["<|endoftext|>"]
# )

# print("Fusions apprises :")

# print (vocab)
# if __name__=="__main__":
#     bottleneck(train_bpe,input_path=r"C:\Users\lenovo\Documents\Cours LLM Stanford\code_apprentissage\data\sample.text",vocab_size=300,special_token=["<|endoftext|>"])

if __name__ == "__main__":

    vocab,_ = train_bpe(
        input_path="sample.text",
        vocab_size=50000,
        special_token=["<|endoftext|>"],
        num_processes=8,
    )

    print(vocab)




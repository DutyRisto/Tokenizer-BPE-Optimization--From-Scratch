import os 
from typing  import BinaryIO


def find_chunk_boundaries(
        file :BinaryIO,
        desired_num_chunks : int ,
        split_special_token : bytes,
        *args,**kwargs
    )->list[int]: 

    assert isinstance(split_special_token,bytes)# Must represent special token as a bytes

    # Cursor position  
    # total size  like len but len not a function of BinaryIO format or ByteIO
    file.seek(0,os.SEEK_END)#end  [move cursor at the end ]
    file_size= file.tell()# position (where the cursor) # give the lenght of octets in the file 

    file.seek(0)

    # initialisation
    chunk_size= file_size // desired_num_chunks
    chunk_boundaries=[i * chunk_size for i in range(desired_num_chunks+1)]
    chunk_boundaries[-1]=file_size

    mini_chunk_size=4096

    for bi in range(1,len(chunk_boundaries)-1):
        start = chunk_boundaries[bi]
        file.seek(start)

        while True: 
            # read octet by octet  each 4k 
            
            mini_chunk= file.read(mini_chunk_size)

            # if the EOF end of File

            if mini_chunk==b'':
                chunk_boundaries[bi]=file_size
                break
            # if found special token 
            found_at = mini_chunk.find(split_special_token) # give the position if have a match,  return -1 not match , int >=0 if found 
            if found_at !=-1:
                chunk_boundaries[bi]=start+found_at
                break
            # if not found special token 
            start+=mini_chunk
    return sorted(set(chunk_boundaries))

            

# # usage

# with open(input_path,'rb') as f:
#     boundaries=find_chunk_boundaries(
#         f,
#         desires_num_chunks
#         split_special_token
#     )

#     for start, end in zip(boundaries[:-1],boundaries[1:]):
#         f.seek(start)
#         chunk=f.read(end-start).decode(encoding='utf-8',erros='ignore')
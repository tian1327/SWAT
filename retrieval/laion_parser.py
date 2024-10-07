import sqlite3
import os
import pyarrow.parquet as pq
import multiprocessing
from multiprocessing import Pool, set_start_method
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse
import json
import pickle
# import queue
import re
import gc
import yaml

"""
1. Normalize text.
2. Find alternate names for labels with single names in imagenet.
3. Next, normalized search for both human engineered, and original Imagenet-1K labels.
"""
NUM_TABLES = {
        'LAION400M.db':  32,
        'LAION2B.db': 128
}


class LaionParser():
    def __init__(self, database, mode='single', 
                 data_source= './', max_threads:int = 16, 
                 num_proc: int = 32, prefix: str = None,
                 matching_strategy: str = 'RELAXED') -> None:
        self.database = database
        self.mode = mode
        self.data_source = data_source
        if self.mode == 'single':
            self.conn = self.__connect__()
        self.tables = {
            'LAION400M.db':  32,
            'LAION2B.db': 128
        }
        self.start_time = time.time()
        self.num_tables = NUM_TABLES[self.database]
        self.max_theads = max_threads
        self.num_proc = min(num_proc, multiprocessing.cpu_count())
        self.prefix = prefix
        self.matching_strategy = matching_strategy
    
    # Needed only once. Don't execute later.
    def create_table(self, shard):
        # print(f'Creating Dataframe and shard. - {str(shard).zfill(5)}')
        shard_id = str(shard).zfill(5)
        parquet_file = pq.ParquetFile(f'./laion2b/part-{shard_id}-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet')
        df = parquet_file.read().to_pandas()
        df.to_sql(f'part{shard}', self.conn)
        self.conn.commit()
        return df
    
    def find_by_id(self, rowid: str, shard: int, column: str):
        cursor = self.conn.cursor()
        cursor.execute(f'''
            select {column}, nsfw from part{shard} where rowid = {rowid};
        ''')
        result = cursor.fetchone()
        return result
    
    # Needed only once. Don't execute later.
    def create_fts_table(self, df, shard):
        print(f'Creating FTS table - {shard}')
        cursor = self.conn.cursor()
        texts = df['TEXT'].tolist()
        # Normalize text and save.
        texts_norm = [text.replace('"', '').replace("'", '').replace("-", ' ').replace('_',' ') if text else "N.A." for text in texts]
        texts_norm = [tuple([text],) for text in texts_norm]
        cursor.execute(f'''
            CREATE VIRTUAL TABLE _fts{shard} USING FTS5(TEXT);
        ''') 
        self.conn.commit() 
        cursor.executemany(f"INSERT INTO _fts{shard} (TEXT) values(?)", texts_norm)
        self.conn.commit()
    
    def get_label_stats(self, label, shard, cursor):
        label = label
        parsed_label = self.parse_multi_words(label)
        if ("(" in label and ")" in label) or "." in label or '/' in label:
            cursor.execute(f'''
                select rowid, text from _fts{shard} where _fts{shard} MATCH '"{parsed_label}"' ORDER BY RANK;
            ''')
        else:
            cursor.execute(f'''
                select rowid, text from _fts{shard} where _fts{shard} MATCH '{parsed_label}' ORDER BY RANK;
            ''')
        matches = cursor.fetchall()
        return matches
    
    def parse_multi_words(self, text: str):
        text = clean_text(text=text)
        text = text.replace(" ", " + ") # replace spaces with + 
        if "(" in text and ")" in text:
            text = text.replace("(", "").replace(")", "")
        if self.prefix is not None:
            text = self.prefix +" "+ text
        return text

    def __connect__(self):
        return sqlite3.connect(os.path.join(self.data_source, self.database))

    def __get_frequency_worker__(self, args):
        key, metrics = args
        names = list(metrics['alternates'].keys())
        conn = self.__connect__()
        cursor = conn.cursor()
        all_matches = 0
        for name in names:
            # if metrics['alternates'][name] != 0: continue
            total_matches = 0
            for i in range(self.num_tables):
                total_matches += self.get_label_stats(name, shard=i, cursor=cursor)
            # with ThreadPoolExecutor(8) as executor:
            #     matches = list(executor.map(lambda item: self.get_label_stats(name, item), range(self.num_tables)))
            # total_matches += sum(matches)
            all_matches += total_matches
            metrics['alternates'][name] = total_matches
            # print(f'all matches - {all_matches} - till {name} - {key}')
        conn.close()
        print(f'Frequency counted for: {key} -  time: {time.time()-self.start_time} - matches: {all_matches}')
        return {key: metrics}
        

    def __get_text_worker__(self, args):

        conn = self.__connect__()
        cursor = conn.cursor()
        key, metrics = args
        # sorted_metrics = sorted(metrics['synonyms_final'].items(), key=lambda x: x[1])
        sorted_metrics = sorted(metrics['alternates'].items(), key=lambda x: x[1])

        label_stack = [item[0] for item in sorted_metrics] # this is the alternative names
        searched = set()
        if 'changed_name' in metrics:
            most_common_name = metrics['changed_name']
            label_stack.append(most_common_name) 
        
        print('label stack', label_stack)
        total_matches = set()
        caption_set = set()
        while len(label_stack) != 0:
            name = label_stack.pop()
            og_name = "".join(name)
            name = clean_text(name)
            # print('name', name)
            # print('og_name', og_name)

            if name in searched: # skip if already searched.
                continue
            searched.add(name)
            name_count = 0
            name = self.parse_multi_words(name) # +++++ The " + " operator between tokens means that a row will match if it contains all of the tokens.
            try:
                for shard in range(self.num_tables):
                    if ("(" in name and ")" in name) or "." in name or '/' in name:
                        cursor.execute(f'''
                            select rowid, text from _fts{shard} where _fts{shard} MATCH '"{name}"' ORDER BY RANK;
                        ''')
                    else:
                        cursor.execute(f'''
                            select rowid, text from _fts{shard} where _fts{shard} MATCH '{name}' ORDER BY RANK;
                        ''')
                    matches = cursor.fetchall() # a list of tuples, where each tuple contains the rowid and text of a row from the table. Each match is (rowid, text)
                    new_matches = [(shard,)+ match for match in matches]
                    name_count += len(new_matches)
                    for match_i in new_matches:
                        if match_i not in caption_set: # remove duplicated matched captions
                            caption_set.add(match_i)
                            total_matches.add((og_name,)+match_i)
                    # total_matches.update(new_matches)
                # metrics['synonyms_final'][og_name] = name_count # update the count
                metrics['alternates'][og_name] = name_count # update the count                            
            except:
                print('exception', og_name, key)

        conn.close()
        return ({key: metrics}, {key: total_matches})

    def get_freq_parallel(self, metrics):
        metrics_flattened = []
        for item in metrics.items():
            key, val = item
            metrics_flattened.append((key,val))
        # metrics_flattened = list(metrics.items())
        # start = time.time()
        num_processes = min(len(metrics_flattened), 32)
        set_start_method('fork')
        with ThreadPoolExecutor(self.max_theads) as pool:
            futures_to_results = {
                pool.submit(self.__get_frequency_worker__,(key, val)): (key,val) for (key,val) in metrics_flattened
            }
            results = pool.map(self.__get_frequency_worker__, metrics_flattened)

        for result in results:
            metrics.update(result)
        print(f'Total time: {time.time()-self.start_time}')
        return metrics


    def get_text_parallel(self, metrics):

        metrics_flattened = []
        for key, val in metrics.items():            
            metrics_flattened.append((key, val))
        total_class_ct = len(metrics_flattened)
        
        retrieved_captions = {}
        # set_start_method('fork')
        class_frequency = {}
        processed_classes = 0
        with ThreadPoolExecutor(self.max_theads) as pool:
            futures_to_metrics = {
                pool.submit(self.__get_text_worker__, (key, val)): (key,val) for (key,val) in metrics_flattened
            }

            for future in as_completed(futures_to_metrics):
                result = future.result()
                (key, value) = futures_to_metrics.pop(future)
                updated_metrics, matches = result
                metrics.update(updated_metrics)
                retrieved_captions.update(matches)

                metrics[key]['most_common_name'] = find_most_common_name(metrics=metrics[key], matching_strategy=self.matching_strategy) # +++++ 
                metrics[key]['actual_freq'] = len(retrieved_captions[key])

                class_frequency[key] = {}
                class_frequency[key]['name'] = metrics[key]['name']
                class_frequency[key]['actual_freq'] = metrics[key]['actual_freq']

                processed_classes +=1
                print(f'Total processed: {processed_classes}/{total_class_ct} - Processed {key} -> label: {value["name"]}, freq: {metrics[key]["actual_freq"]} - time: {time.time() - self.start_time}')
               
                gc.collect()

            # results = pool.map(self.__get_text_worker__, metrics_flattened)

        # for result in results:
        #     updated_metrics, matches = result
        #     metrics.update(updated_metrics)
        #     retrieved_captions.update(matches)
        # for key in metrics.keys():
        #     metrics[key]['most_common_name'] = find_most_common_name(metrics=metrics[key])
        #     metrics[key]['actual_freq'] = len(retrieved_captions[key])
        #     if metrics[key]['actual_freq'] < 100:
        #         class_frequency.append(f"name: {metrics[key]['name']} class: {key} freq: {len(retrieved_captions[key])}")
        
        # sort the class frequency by actual frequency.
        class_frequency = dict(sorted(class_frequency.items(), key=lambda x: x[1]['actual_freq'], reverse=True))
        
        return retrieved_captions, metrics, class_frequency

def clean_text(text: str):
    return text.strip().replace("'",'').replace('"','').replace('-', ' ').replace('_', ' ').replace("  ", ' ').lower()


def find_most_common_name(metrics:dict, matching_strategy:str = 'RELAXED'):

    if 'changed_name' in metrics:
        official_name = metrics['changed_name'] # red eft rather than eft, manual tweaks for some corner cases
    else:
        official_name = metrics['name'] # this is the class name
    official_name_og = "".join(official_name)

    # Order from the official name.
    # alternates_ordered = dict(sorted(metrics['synonyms_final'].items(), key=lambda x: x[1], reverse=True)) # {official_name_og: metrics['alternates'][official_name_og]}
    alternates_ordered = dict(sorted(metrics['alternates'].items(), key=lambda x: x[1], reverse=True)) # {official_name_og: metrics['alternates'][official_name_og]}
    most_common_name = "".join(official_name)

    if official_name not in alternates_ordered:
        clean_official_name = clean_text(official_name)
        if clean_official_name in alternates_ordered:
            most_common_name_freq = alternates_ordered[clean_official_name]
        else:
            most_common_name_freq = 0
    else:
        most_common_name_freq = alternates_ordered[official_name]

    official_name = clean_text(official_name)
    official_name = re.sub(r'[^\w\s]', '', official_name)
    official_name_split = set(official_name.split())
    
    for alternate_name in alternates_ordered.keys():
        
        alternate_name_freq = alternates_ordered[alternate_name]
        alternate_name_og = "".join(alternate_name)
        alternate_name = clean_text(alternate_name)
        alternate_name = re.sub(r'[^\w\s]', '', alternate_name) 
        alternate_name_split = set(alternate_name.split())

        # the 2 relaxed conditions below are to handle the case when the alternate names is a subset of the official name.
        # e.g. official name = "green lacewing", alternate name = "lacewing". 
        # The alternate name is a subset of the official name, and must have larger freq.
        # however, we want to keep the official name as the most common name, because subset name "lacewing" is too generic.
        if most_common_name_freq < alternate_name_freq:
            if (matching_strategy == 'STRICT'):
                most_common_name = alternate_name_og
            elif matching_strategy == 'RELAXED' and alternate_name_split == official_name_split: # This can only happen - Honda Accord 2012 vs 2012 Honda Accord
                most_common_name = alternate_name_og
            elif matching_strategy == 'RELAXED' and not alternate_name_split.issubset(official_name_split): # official name = ""
                most_common_name = alternate_name_og
            most_common_name_freq = alternate_name_freq # subset will generally have a higher freq.
                
    if official_name_og != most_common_name:
        print(f'{official_name_og} - most common name is: {most_common_name}')

    return most_common_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--database', type=str, default='LAION400M', help='Database to mine.')
    parser.add_argument('--dataset', type=str, default='imagenet_1k', help='Downstream dataset.')
    parser.add_argument('--root', type=str, default='', help='Root directory for storing mined data.')
    parser.add_argument('--datasource', type=str, default='database/', help='Location where DB is.')
    parser.add_argument('--max_threads', type=int, default=16, help='Max number of threads to spawn.')
    parser.add_argument('--num_proc', type=int, default=16, help='Number of processes to query Sqlite')
    parser.add_argument('--prefix', type=str, default=None, help='Prefix for datasets like EuroSAT, DTD.')
    parser.add_argument('--matching_strategy', type=str, default='RELAXED', choices=['RELAXED', 'STRICT'])
    parser.add_argument('--tag', type=str, default=None, help='Tag for the dataset.')
    args = parser.parse_args()
    database = f'{args.database}.db'
    
    # read the retrieved path from the config.yml
    with open('../config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.root = config['retrieved_path']         

    laion_parser = LaionParser(database, mode='parallel', 
                               data_source=args.datasource, 
                               max_threads=args.max_threads, 
                               num_proc=args.num_proc,
                               prefix=args.prefix,
                               matching_strategy=args.matching_strategy)   

    if args.dataset == 'semi-aves':
        fn = f'../data/{args.dataset}/{args.dataset}_metrics-LAION400M_query.json' # this file contains more names queried from the ChatGPT.
        # fn = f'dataset/semi-aves/semi-aves_metrics-LAION400M.json'
        # fn = f'dataset/{args.dataset}/{args.dataset}_synonyms_filtered_final_manualcheck.json'

    elif args.dataset == 'stanfordcars': # for cars use corase for retrieval
        fn = f'../data/{args.dataset}/{args.dataset}_metrics-LAION400M-coarse.json'    
    else:
        fn = f'../data/{args.dataset}/{args.dataset}_metrics-LAION400M.json'
    # fn = f'../data/{args.dataset}/{args.dataset}_metrics-LAION400M.json'
    print(f'metric file: {fn}')

    metrics = json.load(open(fn, 'r'))

    output_path = f'{args.root}/{args.dataset}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f'Created directory: {output_path}')
    
    start = time.time()
    mined_captions, metrics, class_frequency = laion_parser.get_text_parallel(metrics) # +++++
    print(f'Total string macthing time: {round((time.time()-start)/60, 0)} minutes')

    ## Save the mined captions and metrics.
    file_name_tag = f'{args.database}'
    if args.tag is not None:
        file_name_tag += f'-{args.tag}'
    if args.prefix is not None:
        file_name_tag += f'-{args.prefix}'

    fn = f'{args.root}/{args.dataset}/{args.dataset}_mined_captions-{file_name_tag}.pkl'
    with open(fn, 'wb') as f:
        pickle.dump(mined_captions, file=f)
    print(f'Saved mined captions: {fn}')

    fn = f'{args.root}/{args.dataset}/{args.dataset}_metrics-{file_name_tag}.json'
    with open(fn, 'w') as f:
        f.write(json.dumps(metrics, indent=4))
    print(f'Saved metrics files: {fn}')

    fn = f'{args.root}/{args.dataset}/{args.dataset}_class_frequency-{file_name_tag}.json'
    with open(fn, 'w') as f:
        f.write(json.dumps(class_frequency, indent=4))
    print(f'Saved class frequency files: {fn}')

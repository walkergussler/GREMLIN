import argparse, os
from calculate_features import wrapper_main as feature_getter
from automatic_feature_selection import build_model as feature_selector

def main(dir_0,dir_1,full_file,output,save_intermediate):
  status_dic={}
  for file in os.listdir(dir_0):
    status_dic[file]=0
  for file in os.listdir(dir_1):
    status_dic[file]=1
  big_data=feature_getter(status_dic,dir_0,dir_1)
  small_data=feature_selector(big_data)
  small_data.to_csv(output+'.csv')
  if save_intermediate:
    big_data.to_csv(output+'_full.csv')

if __name__=="__main__":
  parser=argparse.ArgumentParser(description="Wrapper for GREMLIN for feature engineering and automated machine learning")
  parser.add_argument('-i0', '--dir_0', 
    type=str, required=False, default='./samples/recent/', 
    help="Folder with your samples from your '0' class")
  parser.add_argument('-i1', '--dir_1', 
    type=str, required=False, default='./samples/non-recent/', 
    help="Folder with your samples from your '1' class")
  parser.add_argument('-f', '--full_file', #is this bugged? may not work
    action='store_true', default=False, 
    help="Add this flag to process the whole file rather than the largest 1-step connected component. Turning this option on reduces final model accuracy for the HCV recency problem.")
  parser.add_argument('-i', '--save_intermediate', 
    action='store_true', default=True, 
    help="Add this flag to save the larger file with the large feature set calculated.")
  parser.add_argument('-o', '--output', 
    type=str,required=False, default="final_model", 
    help="Path and name for output file. Do not include extension, output format is .csv")
  args=parser.parse_args()
  main(args.dir_0,args.dir_1,args.full_file,args.output,args.save_intermediate)
def normalize_vec(v):
    """Force values into [0,1] range and replace nan values with mean"""
    # print(v)
    x=max(v)
    m=min(v)
    d=x-m
    out=[]
    n=np.nanmean(v)
    for i in v:
        # print('---')
        # print(i)
        if np.isnan(i):
            val=(n-m)/d
        else:
            if i=='inf':
                val=1
            else:
                val=(i-m)/d
        out.append(val)
    return out

# def make_np_array(d):
    # rownum=len(d)
    # colnum=len(d[0])
    # out=np.zeros([rownum,colnum])
    # print(np.shape(out))
    # for i in range(rownum):
        # row=d[i]
        # out[i,:]=row
    # return out
    
import numpy as np
import sys
statusdic={'APA_new': 0, 'HCDE18': 0, 'C13ND19_new': 0, 'WA6': 0, 'HOC_P07_1b_new': 0, 'LBA_new': 0, 'C13ND21_new': 0, 'C13ND123_new': 0, 'C13ND132_new': 0, 'HOC_P09_1b_new': 0, 'C13ND6_new': 0, 'RAM_new': 0, 'C13ND142_new': 0, 'HCDE10': 0, 'WA9': 0, 'C13ND31_new': 0, 'MVA_new': 0, 'EEV_new': 0, 'WA11': 0, 'WA8': 0, 'WA14': 0, 'HOC_P21_1b_new': 0, 'C13ND126_new': 0, 'C13ND16_new': 0, 'DGO_new': 0, 'C13ND14_new': 0, 'DHE_new': 0, 'C13ND32_new': 0, 'LRA_new': 0, 'C13ND70_new': 0, 'HOC_P16_1b_new': 0, 'NCH_new': 0, 'HOC_P28_1b_new': 0, 'HOC_P20_1b_new': 0, 'HOC_P12_1b_new': 0, 'C13ND10_new': 0, 'CCI_new': 0, 'MRU_new': 0, 'WA2': 0, 'C13ND4_new': 0, 'MSMPAN4_new': 0, 'C13ND20_new': 0, 'WA10': 0, 'C13ND121_new': 0, 'C13ND140_new': 0, 'HOC_P22_1b_new': 0, 'TRO_new': 0, 'JSA_new': 0, 'SFGH008_new': 0, 'C13ND22_new': 0, 'JJO_new': 0, 'HOC_P01_1b_new': 0, 'HCDE9': 0, 'MSMPAN16_new': 0, 'VVI_new': 0, 'C13ND33_new': 0, 'CBA_new': 0, 'HOC_P13_1b_new': 0, 'AHCV03_new': 0, 'HCDE4': 0, 'HOC_P04_1b_new': 0, 'C13ND170_new': 0, 'C13ND34_new': 0, 'C13ND15_new': 0, 'HOC_P10_1b_new': 0, 'TKE_new': 0, 'HCDE14': 0, 'HOC_P08_1b_new': 0, 'WA7': 0, 'C13ND130_new': 0, 'MSMPAN19_new': 0, 'JOP_new': 0, 'RMA_new': 0, 'C13ND173_new': 0, 'SFGH005_new': 0, 'MSMPAN3_new': 0, 'WA5': 0, 'HCDE5': 0, 'C13ND7_new': 0, 'WA13': 0, 'WA19': 0, 'WA17': 0, 'MSMPAN15_new': 0, 'C13ND117_new': 0, 'C13ND129_new': 0, 'HCDE2': 0, 'CGO_new': 0, 'C13ND11_new': 0, 'C13ND162_new': 0, 'C13ND18_new': 0, 'AHCV08_new': 0, 'HCDE20': 0, 'HOC_P27_1b_new': 0, 'C13ND2_new': 0, 'HOC_P18_1b_new': 0, 'MSMPAN11_new': 0, 'C13ND120_new': 0, 'C13ND25_new': 0, 'C13ND172_new': 0, 'HCDE17': 0, 'SAL_new': 0, 'C13ND125_new': 0, 'HCDE8': 0, 'C13ND24_new': 0, 'ATA_new': 0, 'C13ND12_new': 0, 'C13ND23_new': 0, 'WA3': 0, 'C13ND128_new': 0, 'C13ND127_new': 0, 'HOC_P34_1b_new': 0, 'HCDE1': 0, 'HCDE11': 0, 'HCDE13': 0, 'C13ND1_new': 0, 'C13ND17_new': 0, 'C13ND131_new': 0, 'C13ND5_new': 0, 'NH3320_new': 0, 'C13ND27_new': 0, 'KOM_P141': 1, 'LYB_P03': 1, 'AMC_P31': 1, '24GLC': 1, 'VAO_P30': 1, '12GLC': 1, '18GLC': 1, 'LYB_P59': 1, 'BID_P14T1': 1, 'VAO_P01': 1, 'KOM_P215': 1, 'LYB_P60': 1, 'AMC_P47': 1, 'KOM_P146': 1, 'SGH54': 1, 'KOM_P059': 1, 'LYB_P64': 1, 'KOM_P062': 1, 'PAK_P10': 1, 'KOM_P065': 1, '14GLC': 1, 'SGH36': 1, 'BID_P11T1': 1, 'SGH34': 1, 'PAK_P13': 1, 'VAO_P39': 1, '16GLC': 1, 'KOM_P022': 1, 'BID_P12T1': 1, 'BID_P02T1': 1, 'LYB_P39': 1, 'VAO_P31': 1, 'KOM_P060': 1, 'LYB_P57': 1, 'AMC_P06': 1, 'AMC_P01': 1, 'BID_P03T1': 1, 'KOM_P080': 1, 'LYB_P06': 1, 'KOM_P188': 1, 'VAO_P52': 1, 'PAK_P08': 1, 'VAO_P25': 1, 'KOM_P297': 1, 'VAO_P29': 1, 'SGH23': 1, 'PAK_P19': 1, 'VAO_P51': 1, 'KOM_P235': 1, 'LYB_P48': 1, 'AMC_P18': 1, 'KOM_P066': 1, 'KOM_P250': 1, 'VAO_P15': 1, 'BID_P04T1': 1, 'SGH25': 1, 'AMC_P43': 1, 'KOM_P222': 1, 'SGH49': 1, 'KOM_P163': 1, 'VAO_P06': 1, 'BID_P01T1': 1, '01GLC': 1, 'BID_P06T1': 1, 'LYB_P01': 1, 'AMC_P66': 1, 'KOM_P004': 1, '10GLC': 1, 'KOM_P229': 1, 'KOM_P257': 1, 'AMC_P44': 1, 'LYB_P15': 1, 'VAO_P18': 1, 'SGH33': 1, 'KOM_P236': 1, 'AMC_P57': 1, 'VAO_P27': 1, 'VAO_P49': 1, 'VAO_P35': 1, 'SGH50': 1, 'PAK_P01': 1, 'SGH40': 1, 'AMC_P24': 1, 'AMC_P46': 1, 'KOM_P135': 1, 'LYB_P30': 1, 'VAO_P37': 1, 'AMC_P58': 1, 'PAK_P12': 1, '06GLC': 1, 'VAO_P07': 1, 'KOM_P205': 1, 'VAO_P11': 1, 'LYB_P45': 1, 'KOM_P199': 1, 'LYB_P50': 1, 'LYB_P14': 1, 'LYB_P51': 1, 'AMC_P45': 1, 'AMC_P70': 1, 'LYB_P33': 1, 'VAO_P26': 1, 'AMC_P12': 1, '03GLC': 1, 'AMC_P02': 1, 'KOM_P218': 1, 'VAO_P40': 1, 'AMC_P42': 1, 'SGH35': 1, 'PAK_P11': 1, 'BID_P15T1': 1, 'KOM_P221': 1, 'VAO_P32': 1, 'VAO_P02': 1, 'LYB_P28': 1, 'KOM_P069': 1, 'KOM_P044': 1, '21GLC': 1, 'KOM_P039': 1, 'AMC_P04': 1, '04GLC': 1, '25GLC': 1, 'LYB_P66': 1, 'AMC_P16': 1, 'AMC_P62': 1, 'KOM_P246': 1, 'VAO_P14': 1, 'LYB_P54': 1, 'LYB_P35': 1, 'BID_P13T1': 1, 'LYB_P27': 1, 'SGH51': 1, 'KOM_P248': 1, 'KOM_P359': 1, 'LYB_P38': 1, 'LYB_P52': 1, 'VAO_P46': 1, 'KOM_P291': 1, 'KOM_P011': 1, 'VAO_P16': 1, 'VAO_P50': 1, 'AMC_P05': 1, 'LYB_P11': 1, 'AMC_P14': 1, '23GLC': 1, 'PAK_P18': 1, 'KOM_P084': 1, 'SGH39': 1, 'KOM_P033': 1, 'PAK_P14': 1, 'SGH31': 1, 'SGH26': 1, 'KOM_P003': 1, 'KOM_P201': 1, 'VAO_P13': 1, 'KOM_P134': 1, 'SGH44': 1, 'SGH41': 1, 'PAK_P07': 1, 'LYB_P26': 1, 'VAO_P05': 1, 'KOM_P061': 1, 'KOM_P217': 1, 'KOM_P170': 1, '08GLC': 1, '22GLC': 1, 'KOM_P220': 1, 'AMC_P32': 1, 'LYB_P62': 1, 'SGH52': 1, 'AMC_P39': 1, 'KOM_P177': 1, 'SGH27': 1, '02GLC': 1, 'VAO_P48': 1, 'VAO_P24': 1, 'VAO_P08': 1, 'KOM_P362': 1, '15GLC': 1, 'VAO_P09': 1, 'AMC_P49': 1, '09GLC': 1, 'SGH55': 1, 'LYB_P61': 1, 'VAO_P45': 1, 'LYB_P46': 1, 'PAK_P06': 1, 'AMC_P15': 1, 'KOM_P063': 1, 'VAO_P12': 1, 'AMC_P37': 1, 'KOM_P046': 1, '11GLC': 1, 'LYB_P53': 1, 'VAO_P44': 1, 'AMC_P09': 1, 'AMC_P03': 1, 'VAO_P28': 1, 'LYB_P31': 1, 'AMC_P13': 1, 'KOM_P233': 1, 'VAO_P19': 1, 'KOM_P032': 1, 'LYB_P07': 1, 'LYB_P40': 1, 'KOM_P203': 1, 'KOM_P204': 1, 'AMC_P38': 1, 'LYB_P63': 1, 'AMC_P35': 1, 'LYB_P44': 1, 'AMC_P08': 1, 'LYB_P05': 1, 'KOM_P234': 1, 'KOM_P241': 1, 'KOM_P227': 1, 'VAO_P54': 1, 'LYB_P41': 1, 'VAO_P22': 1, 'SGH42': 1, 'KOM_P354': 1, 'PAK_P05': 1, 'BID_P05T1': 1, 'VAO_P36': 1, 'SGH47': 1, 'PAK_P04': 1, 'LYB_P58': 1, 'KOM_P085': 1, 'KOM_P111': 1, 'LYB_P37': 1, 'LYB_P22': 1, '07GLC': 1, 'BID_P10T1': 1, 'PAK_P02': 1, 'VAO_P41': 1, 'LYB_P32': 1, '19GLC': 1, '17GLC': 1, 'LYB_P20': 1, 'VAO_P43': 1, 'AMC_P53': 1, 'AMC_P63': 1, 'VAO_P20': 1, 'AMC_P51': 1, 'SGH30': 1, 'VAO_P17': 1, 'VAO_P21': 1, 'AMC_P48': 1, 'AMC_P30': 1, 'SGH29': 1, 'KOM_P239': 1, 'VAO_P53': 1, 'LYB_P43': 1, 'SGH46': 1, 'AMC_P40': 1, 'SGH43': 1, 'KOM_P154': 1}
d=[]
files=[]
c=0
GROUP_VAL=2
with open(sys.argv[1]) as f:
    for line in f.readlines():
        c+=1
        s=line.strip().split(',')
        if c>1:
            data=list(map(float,s[1:]))
            d.append(data)
            files.append(s[0])
        else:
            linelen=len(s)-1
            print(line.strip()+',status')
arr=np.array(d) #why doesnt this work?
# arr=np.matrix(d)
# arr=make_np_array(d)
# print(np.shape(arr))
for i in range(linelen):
    # print(i)
    w=arr[:,i]
    norm_w=normalize_vec(w)
    arr[:,i]=norm_w
for i in range(len(arr)):
    file=files[i]
    fixed=file.replace('_onestep','')
    s=list(arr[i,:])
    if fixed in statusdic:
        s.append(statusdic[fixed])
    else:
        s.append(2)
    print(fixed+','+','.join(map(str,s)))

import os
omniglot_path=os.getcwd()+r'\datas\omniglot_resized'
alpha_families=[omniglot_path+'\\'+file for file in os.listdir(omniglot_path)]

characters=[file for alpha_family in alpha_families for file in os.listdir(alpha_family) if os.path.isdir(alpha_family+'\\'+file)]
print(len(characters))
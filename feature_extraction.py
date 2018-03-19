import tensorflow as tf 
import os
import pickle
import numpy as np 
from inceptionv1 import *
from pprint import pprint

def feature_extract(pickledir, savedir, pre_params_dir):
	savedir=os.path.join(savedir, 'feature_30')
	if not os.path.isdir(savedir):
		os.mkdir(savedir)
		os.mkdir(savedir+"//0")
		os.mkdir(savedir+"//1")

	len_input=224*224*3 #input 길이
	dictlyr=np.load(pre_params_dir, encoding='latin1').item() #googlenet.npy열기, latin1타입
	params_pre=reformat_params(dictlyr) #W,b parameter dictionary로 만들기

	X=tf.placeholder(tf.float32, [None, len_input])
	feature=arxt(X, params_pre) # arxt : conv layer topology를 타는 과정

	merge_pat_list_0=os.listdir(pickledir + "//0") #pickle list 만들기
	merge_pat_list_0=sorted(merge_pat_list_0) #그리고 sorting 하기

	merge_pat_list_1 = os.listdir(pickledir + "//1")
	merge_pat_list_1 = sorted(merge_pat_list_1)

	# for CT_30 in merge_pat_list_0:
	# 	pat_path = os.path.join(pickledir ,"0" , CT_30 )
	# 	with tf.Session() as sess:
	# 		init=tf.global_variables_initializer()
	# 		sess.run(init)
    #
	# 		try :
	# 			append_array = np.zeros(shape = (30,1024))
	# 			for i in range(30):
	# 				each_pat_CT_path = os.path.join(pat_path , os.listdir(pat_path)[i])
	# 				with open(each_pat_CT_path, 'rb') as f :
	# 					each_CT = pickle.load(f)
	# 				each_CT = each_CT.reshape(-1, len(each_CT))
	# 				# print(each_CT.shape)
	# 				each_CT_feature = sess.run(feature, feed_dict = {X: each_CT})
    #
	# 				append_array[i] = each_CT_feature
    #
    #
	# 			append_array = append_array.reshape(30*1024)
	# 			with open(savedir + "//0//" + CT_30 + ".pickle", 'wb') as f :
	# 				pickle.dump(append_array, f)
    #
	# 			print(CT_30 +" patient feature is generated." )
	# 		except IndexError:
	# 			pass
	for CT_30 in merge_pat_list_1:
		pat_path = os.path.join(pickledir ,"1" , CT_30 )
		with tf.Session() as sess:
			init=tf.global_variables_initializer()
			sess.run(init)

			append_array = np.zeros(shape = (30,1024))
			try :
				for i in range(30):
					each_pat_CT_path = os.path.join(pat_path , os.listdir(pat_path)[i])
					with open(each_pat_CT_path, 'rb') as f :
						each_CT = pickle.load(f)
					each_CT = each_CT.reshape(-1, len(each_CT))
					# print(each_CT.shape)
					each_CT_feature = sess.run(feature, feed_dict = {X: each_CT})
					append_array[i] = each_CT_feature

				append_array = append_array.reshape(30 * 1024)
				with open(savedir + "//1//" + CT_30 + ".pickle", 'wb') as f :
					pickle.dump(append_array, f)

				print(CT_30 +" patient feature is generated." )
			except IndexError :
				pass
if __name__=='__main__':
	feature_extract("C://Users//korea//Desktop//project//sungmo//bin_data//",'C://Users//korea//Desktop//project//sungmo' ,'C://Users//korea//Desktop//project//sungmo//googlenet.npy')


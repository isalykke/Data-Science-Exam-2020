data_folder = "./data/"

for dirpath, _, files in os.walk(data_folder):
        for file in files[0:20]: 
            if file.endswith(".jpg"):
                path = os.path.join(dirpath,file)
                print(path)



from sklearn.model_selection import train_test_split
import numpy

with open("datafile.txt", "rb") as f:
   data = f.read().split('\n')
   data = numpy.array(data)  #convert array to numpy type array

   x_train ,x_test = train_test_split(data,test_size=0.5)


from sklearn.cross_validation import train_test_split 
X_fit, X_eval, y_fit, y_eval= train_test_split( train, target, test_size=0.15, random_state=1 )
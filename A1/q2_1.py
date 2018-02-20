#assignment 1 question 2
#takes a pairwise distance matrix and
#returns the responsibilities of the training examples to a new test data point. 

from q1 import euclidian_distance ;
import tensorflow as tf;
import numpy as np;

# dm = pairwise distance matrix (nxd)
# x = data test point (1xd)
# k = number of nearest neighbours you want
#retrun value = k nearest neighbours
def responsibility(x, dm, k): 
    
    dm = tf.negative(dm);#flip all values to find closest neighbours
    values, indices = tf.nn.top_k(dm, k);

    count = tf.convert_to_tensor(dm.shape[1].value); #number of cols in indices matrix
    offset = tf.transpose(tf.expand_dims(tf.range(0,k*count,count),1)); #creates 1x num of row x1
    indices = offset + indices;
    indices = tf.transpose(indices);
    
    indices = tf.reshape(indices,[tf.convert_to_tensor(indices.shape[0].value*indices.shape[1].value)]);
    zero = tf.Variable(tf.zeros([dm.shape[0]*dm.shape[1]],tf.float32));
    
    fill = tf.fill([indices.shape[0]],tf.constant(1/k));
    #fill = tf.scatter_update(zero, indices , fill);
    #result = tf.reshape(fill,[dm.shape[0],dm.shape[1]]);
    
    with tf.Session() as session:
        session.run(tf.global_variables_initializer());
        session.run(fill);
        #session.run(result);
        print(session.run(offset));
        print(session.run(indices));        

    #return result

if __name__ == "__main__":
    print("main:");
    test = [[0,1,0],[1,1,1]];
    dm = [[0,0,0], [1,1,1], [2,2,2], [3,3,3]];
    print("distanceMatrix:");
    distanceMatrix = euclidian_distance(test,dm);
    result = responsibility(test,distanceMatrix,2);
    print(result);
    

    
    
    
    
    
    
    
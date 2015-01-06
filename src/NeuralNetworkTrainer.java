import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;


/**
 * Trains a neural network without hidden layers. Outputs
 * <fold> <predicted-class> <actual-class> <confidence> to
 * the console.
 * @author Ashwin Karthi Narayanaswamy
 * @version 1.0
 * 
 */

public class NeuralNetworkTrainer {
	
	//global variables
	public static Instances dataSet;
	public static double[] weights;
	public static double[] confidence;
	public static double learningRate;
	public static int[] foldMap;
	public static double avgTestAccuracy, avgTrainAccuracy;
	public static double threshold;
	
	/**
	 * Function to initialize the weights of the neural network to
	 * a default value
	 * @param weights
	 * @param initialWeight
	 * @return void (Changes the weights array values implicitly)
	 */
	public static void setInitialWeight(double initialWeight){
		for(int i = 0; i<weights.length; i++){
			weights[i] = initialWeight;
		}
	}
	
	/**
	 * Does gradient descent for epoch rounds
	 * @param epoch
	 * @return void (Implicitly changes the weights array)
	 */
	public static void doNRoundGD(int epoch, Instances trainingSet){
		for(int i=0;i<epoch;i++){
			doOneRoundGD(trainingSet);
		}
	}
	
	/**
	 * Does a single round of Gradient Descent and updates the
	 * value of the weights
	 */
	public static void doOneRoundGD(Instances trainingSet){
		int i = 0, j = 0;
		double input = 0, predictedop = 0, delta = 0, actualop = 0;
		Instance instance;
		
		for(i=0; i<trainingSet.numInstances();i++){
			
			//for each instance reduce update the weights array
			instance = trainingSet.instance(i);
			
			//predict the output from the value
			predictedop = predictClassValue(instance);
			
			//get the actual output
			actualop = instance.classValue();
			
			//calculate delta
			delta = predictedop*(1-predictedop)*(actualop-predictedop);
			
			//update the biased weight
			input = 1;
			weights[0] = weights[0] + learningRate*delta*input; //accounting for the biased weight
			
			//for each attribute update the corresponding weight in
			//the weights array
			for(j=0; j<trainingSet.numAttributes()-1;j++){
				input = instance.value(j);
				weights[j+1] = weights[j+1] + learningRate*delta*input; //accounting for the biased weight
			}
		}
	}
	
	
	/**
	 * Predicts the class of the given instance by using the weights array
	 * @param instance
	 * @return 0 if it belongs to class 0 or 1 if it belongs to class 1  
	 */
	public static double predictClassValue(Instance instance){
		double classvalue = 0;
		double value = 1*weights[0];		//biased unit weight
		
		for(int i=0;i<instance.numAttributes()-1;i++){
			value += weights[i+1]*instance.value(i);
		}
		
		classvalue = sigmoid(value);
		return classvalue;
	}
	
	/**
	 * Calculates the accuracy of the testSet against the trained values in weights array
	 * @param testSet
	 * @return accuracy of the testSet 
	 */
	public static double getAccuracy(Instances testSet){
		double accuracy = 0, predictedop = 0, actualop = 0;
		int i = 0;
		
		for(i=0; i<testSet.numInstances(); i++){
			
			actualop = testSet.instance(i).classValue();
			
			if(predictClassValue(testSet.instance(i))>threshold)
				predictedop = 1.0; 
			else
				predictedop = 0;
			
			if(actualop == predictedop)
				accuracy++;
		}
		accuracy = accuracy/testSet.numInstances();
		return accuracy;
	}
	
	/**
	 * Calculates the sigmoid of the given value
	 * @param value
	 * @return the sigmoid of value
	 */
	public static double sigmoid(double value){
		return (double)1/(double)((double)1+Math.exp(-value));
	}
	
	/**
	 * Does stratified cross validation with N folds
	 * @param n
	 */
	public static void doStratifiedNFoldCrossValidation(int folds, int epoch){
		
		int i = 0, j = 0;
		double tp = 0, fp = 0;
		Instances testSet, trainingSet;
		Instance instance;
		
		avgTestAccuracy = 0;
		avgTrainAccuracy = 0;
		
		testSet = new Instances(dataSet,0);
		trainingSet = new Instances(dataSet,0);
		
		for(i=0; i<folds; i++){
			
			//get the instances having this particular value of fold from the dataSet
			for(j=0;j<dataSet.numInstances();j++){
				
				instance = dataSet.instance(j);
				
				if(foldMap[j] == i)
					testSet.add(instance);
				else
					trainingSet.add(instance);
			}
			
			//testSet and trainingSet are obtained
			//randomize the trainingSet
			trainingSet.randomize(new Random(System.currentTimeMillis()));
			
			//initialize the weights
			setInitialWeight(0.1);
			
			//do gradientDescent
			doNRoundGD(epoch, trainingSet);
			
			//test accuracy
			avgTestAccuracy += getAccuracy(testSet);
			
			//train accuracy
			avgTrainAccuracy += getAccuracy(trainingSet);
			
			//calculate the tp and fp
			for(j=0;j<testSet.numInstances();j++){
				if(predictClassValue(testSet.instance(j))>threshold){
					//positive
					if(testSet.instance(j).classValue()==1){
						//true
						tp++;
					}
					else{
						//false
						fp++;
					}
				}
			}
			
			//reset the testSet and trainingSet
			testSet.delete();
			trainingSet.delete();
			
			//calculate the tprate and fprate
			
		}
		
		double tpr = 0, fpr = 0;
		int totalpos = 0, totalneg = 0;
		for(j=0;j<dataSet.numInstances();j++){
			if(dataSet.instance(j).classValue()==0)
				totalneg++;
			else
				totalpos++;
		}
		
		tpr = tp/totalpos;
		fpr = fp/totalneg;
		
		/*System.out.print("\nTrue Pos rate = "+tpr);
		System.out.print("\nFalse Pos rate = "+fpr);*/
		
		avgTestAccuracy /= folds;
		avgTrainAccuracy /= folds;
		
	}
	
	/**
	 * Generates a mapping for each instance with the associated fold.
	 * Produces foldMap[i] = foldValue where 'i' is the i-th instance in the dataSet
	 * and foldValue is the fold to which it is associated to.
	 * @param folds 
	 */
	public static void foldInstances(int folds){
		
		int i = 0, j = 0, positiveCount = 0, negativeCount = 0;
		int pinfold = 0, ninfold = 0;
		int remainingPositive = 0, remainingNegative = 0;
		
		ArrayList<Integer> pmap = new ArrayList<Integer>();
		ArrayList<Integer> nmap = new ArrayList<Integer>();
		
		foldMap = new int[dataSet.numInstances()];
		
		//count the number of positive and negative instances
		for(i=0; i<dataSet.numInstances(); i++){	
			if(dataSet.instance(i).classValue()==0)
				negativeCount++;
			else
				positiveCount++;
		}
		
		pinfold = positiveCount/folds;
		remainingPositive = positiveCount%folds;
		
		ninfold = negativeCount/folds;
		remainingNegative = negativeCount%folds;
		
		for(i=0;i<folds;i++){
			for(j=0;j<pinfold;j++){
				pmap.add(i);
			}
			for(j=0;j<ninfold;j++){
				nmap.add(i);
			}
		}
		
		//evenly distribute the remaining positives into the map 
		if(remainingPositive!=0){
			for(j=0;j<remainingPositive;j++)
				pmap.add(j);
		}
		
		//evenly distribute the remaining negatives into the map
		if(remainingNegative!=0){
			for(j=0;j<remainingNegative;j++)
				nmap.add(j);
		}
		
		//now schuffle pmap and nmap
		Collections.shuffle(pmap);
		Collections.shuffle(nmap);
		
		//System.out.println("The p = "+pmap.size()+" ****n = "+nmap.size());
		
		for(i=0;i<dataSet.numInstances();i++){
			Instance instance = dataSet.instance(i);
			if(instance.classValue() == 0){
				//get the fold number from nmap and add it foldsMap
				foldMap[i] = (int)nmap.get(0);
				nmap.remove(0);
			}
			else{
				//get the fold number from pmap and add it to foldsMap
				foldMap[i] = (int)pmap.get(0);
				pmap.remove(0);
			}
		}
		
		/*for(i=0;i<dataSet.numInstances();i++){
			System.out.print("\nFoldMap val for "+i+" = "+foldMap[i]);
		}*/
	}
	
	/**
	 * Prints the output in the required format
	 */
	public static void printOutput(){
		int i = 0;
		Instance instance;
		double pclass = 0;
		
		for(i=0;i<dataSet.numInstances();i++){
			
			instance = dataSet.instance(i);
			
			//fold
			System.out.print("\n"+foldMap[i]+"\t");
			
			//predicted Class
			pclass = predictClassValue(instance);
			
			if(pclass<threshold) //negative class
				System.out.print(dataSet.attribute(dataSet.classIndex()).value(0)+"\t");
			else
				System.out.print(dataSet.attribute(dataSet.classIndex()).value(1)+"\t");
			
			//print the actual class
			System.out.print(instance.stringValue(instance.classIndex())+"\t");
			
			//print the sigmoid value
			System.out.print(pclass);
		}
		
	}
	
	/**
	 * Main method of the function sets the initial weight for the weights array
	 * Sets the learning rate and loads the data from the file and does the 
	 * required number of round of Gradient Descent and trains the Neural Network
	 * 
	 * @param args
	 */
	public static void main(String args[]){
		
		if(args.length!=4){
			System.out.println("Usage: java -jar <jar-file> <folds> <learning-rate> <epoch> <training-file.arff>");
			return;
		}
		
		int folds = Integer.parseInt(args[0]);
		learningRate = Double.parseDouble(args[1]);
		int epoch = Integer.parseInt(args[2]);
		
		threshold = 0.5;
		
		try{
			
			//load the Instances using weka tool
			BufferedReader reader = new BufferedReader(new FileReader(args[3]));
			dataSet = new Instances(reader);
			reader.close();
		}
		catch(Exception e){
			System.out.print("\nError thrown in main function: "+e);
		}
		
		dataSet.setClassIndex(dataSet.numAttributes() - 1);
		
		//create an array for storing the values of all the weights
		weights = new double[dataSet.numAttributes()];
		
		//set the initial weights and learning rate
		setInitialWeight(0.1);
		
		foldInstances(folds);
		
		doStratifiedNFoldCrossValidation(folds,epoch);
		
		printOutput();	
		
		/*System.out.print("\nAverage Train Set Accuracy = "+avgTrainAccuracy);
		System.out.print("\nAverage Test Set Accuracy = "+avgTestAccuracy);*/
	}

}

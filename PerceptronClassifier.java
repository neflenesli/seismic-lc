package learn.lc.core;
import learn.math.util.VectorOps;

import java.util.Arrays;
import java.util.List;

public class PerceptronClassifier extends LinearClassifier {
	
	public PerceptronClassifier(double[] weights) {
		super(weights);
	}
	
	public PerceptronClassifier(int ninputs) {
		super(ninputs);
	}

	/**
	 * A PerceptronClassifier uses the perceptron learning rule
	 * w_i \leftarrow w_i+\alpha(y-h_w(x)) \times x_i 
	 */

	@Override
	public void update(double[] x, double y, double alpha) {
		double h_w = threshold(VectorOps.dot(weights, x));
		for (int i=0; i<weights.length; i++) {
			weights[i] += alpha * (y - h_w) * x[i]; //(AIMA Eq. 18.7): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times x_i
		}
	}

	@Override
	public String trainingReport(List<Example> examples, int stepnum, int nsteps) {
		double accuracy = accuracyCount(examples); //here the accuracy is different than 1-error, it's simply the count of
												  //correct predictions over all the predictions
		return ("" + stepnum + "\t" + accuracy);
	}

	/**
	 * A PerceptronClassifier uses a hard 0/1 threshold.
	 */
	public double threshold(double z) {
		if (z>=0) return 1.0; //hard threshold, there's no transition between 0 and 1
		return 0.0;
	}
	
}

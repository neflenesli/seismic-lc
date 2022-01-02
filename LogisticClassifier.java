package learn.lc.core;
import java.util.List;
import learn.math.util.VectorOps;

public class LogisticClassifier extends LinearClassifier {
	
	public LogisticClassifier(double[] weights) {
		super(weights);
	}
	
	public LogisticClassifier(int ninputs) {
		super(ninputs);
	}

	@Override
	public String trainingReport(List<Example> examples, int stepnum, int nsteps) {
		double correctness = 1.0-squaredErrorPerSample(examples); //1-L2error
		return ("" + stepnum + "\t" + correctness);
	}

	/**
	 * A LogisticClassifier uses the logistic update rule
	 * w_i \leftarrow w_i+\alpha(y-h_w(x)) \times h_w(x)(1-h_w(x)) \times x_i
	 */

	@Override
	public void update(double[] x, double y, double alpha) {
		double h_w = threshold(VectorOps.dot(weights,x));
		for (int i=0; i<weights.length; i++) {
			weights[i] += alpha * (y - h_w) * h_w * (1 - h_w) * x[i]; //(AIMA Eq. 18.8): w_i \leftarrow w_i+\alpha(y-h_w(x)) \times h_w(x)(1-h_w(x)) \times x_i
		}
	}
	
	/**
	 * A LogisticClassifier uses a 0/1 sigmoid threshold at z=0.
	 */
	@Override
	public double threshold(double z) { //sigmoid threshold implies that there are transitional points between 0 and 1
		return 1 / (1 + Math.exp(-z)); // ( 1 / ( 1 + e^-z) )
	}

}

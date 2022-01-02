package learn.lc.core;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import learn.math.util.VectorOps;

public abstract class LinearClassifier {
	
	public double[] weights;
	List<Double> report = new ArrayList<Double>();
	
	public LinearClassifier(double[] weights) {
		this.weights = weights;
	}
	
	public LinearClassifier(int ninputs) {
		this.weights = new double[ninputs];
	}


	public abstract void update(double[] x, double y, double alpha);


	abstract public double threshold(double z);

	public double eval(double[] x) {
		return threshold(VectorOps.dot(this.weights, x));
	}

	
	protected String trainingReport(List<Example> examples, int stepnum, int nsteps) { //will be overridden
		//System.out.println(stepnum + "\t" + accuracy(examples));
		return "";
	}
	
	public double squaredErrorPerSample(List<Example> examples) { //L2 error
		double sum = 0.0;
		for (Example ex : examples) {
			double result = eval(ex.inputs);
			double error = ex.output - result;
			sum += error*error;
		}
		return sum / examples.size();
	}


	public double accuracyCount(List<Example> examples) {
		int correctCount = 0; //we'll count the correct predictions
		for (Example ex : examples) {
			double result = eval(ex.inputs);
			if (result == ex.output) {
				correctCount += 1;
			}
		}
		return (double)correctCount / examples.size(); //basic proportion here
	}

	public void train(List<Example> examples, int steps, LearningRateSchedule schedule) throws IOException {
		Random rand = new Random();
		int n = examples.size();
			FileWriter writer = new FileWriter(new File("output.csv"));
			BufferedWriter buffWriter = new BufferedWriter(writer);
			for (int i=1; i <= steps; i++) { //writes the training report into an output file in a loop
				int j = rand.nextInt(n);
				Example ex = examples.get(j);
				this.update(ex.inputs, ex.output, schedule.alpha(i)); //classifier-specific update happens here
				String report = this.trainingReport(examples, i,  steps);
				System.out.println(report);
				String[] reportSplit = report.split("\t");
				if (steps>10000) {
					if (i%200==0) { //for high step numbers it takes every 200th data point into consideration
						buffWriter.write(reportSplit[1]);
						buffWriter.newLine();
					}
				} else {
					buffWriter.write(reportSplit[1]); //for low step numbers every point is considered as usual
					buffWriter.newLine();
				}
			}
			buffWriter.close();
	}

	public void train(List<Example> examples, int nsteps, double alpha) throws IOException {
		train(examples, nsteps, new LearningRateSchedule() {
			public double alpha(int t) {
				return alpha; //for the case where the alpha is specified by the user
			}
		});
	}

	public double product(double weight[], double input[]) { //dot product of two vectors -- check
		double result = 0;
		int wsize = weight.length;
		int isize = input.length;
		if (wsize != isize) {
			return 0;
		}
		for (int i = 0; i < wsize; i++) {
			result += weight[i] * input[i];
		}
		return result;
	}
}

package learn.lc.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PerceptronMain {
    public static List<Example> fileReader(String input) throws IOException {

        List<Example> examples = new ArrayList<Example>();
        File fileIn = new File(input);
        BufferedReader bufferReader = new BufferedReader(new FileReader(fileIn));
        String inputLine;

        String[] splitLine = null;
        int size = 0;
        int no_input = 0;

        while ((inputLine = bufferReader.readLine()) != null) { //while there are still lines to read
            splitLine = inputLine.split(",");
            size = splitLine.length;
            no_input = size - 1;
            Example ex = new Example(size);
            ex.output = Double.parseDouble(splitLine[no_input]);
            for (int i = 1; i < size; i++) {
                ex.inputs[i] = Double.parseDouble(splitLine[i - 1]);
            }
            ex.inputs[0] = 1;
            examples.add(ex);

        }
        bufferReader.close();
        return examples;

    }

    public static void learn(String filename, int nsteps, double alpha) throws IOException {
        List<Example> examples = fileReader("learn/examples/" + filename);
        int no_inputs = examples.get(0).inputs.length;
        PerceptronClassifier p = new PerceptronClassifier(no_inputs);

        if (alpha > 0) {  //if the alpha is specified
            p.train(examples, nsteps, alpha);
        } else { //if the alpha isn't specified (decaying alpha)
            p.train(examples, 100000, new DecayingLearningRateSchedule());
        }
    }

    public static void main(String args[]) throws IOException {
        String file = args[0];
        int steps = Integer.parseInt(args[1]);
        double alpha = Double.parseDouble(args[2]);
        learn(file, steps, alpha);
    }
}


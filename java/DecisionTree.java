import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A decision tree classifier
 */
public class DecisionTree implements Classifier {
	
	public enum SplittingRule { INFO_GAIN, RANDOM };

	private SplittingRule rule;
	private int depthLimit;

	
	public DecisionTree(SplittingRule rule, int depthLimit) {
		this.rule = rule;
		this.depthLimit = depthLimit;
	}
	
	public void train(List<Example> examples) {
		//TODO: implement decision tree training
	}
	
	public boolean predict(boolean[] example) {
		// TODO: implement decision tree prediction
		return false;
	}
}

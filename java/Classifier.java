import java.util.List;

/**
 * A common interface for classification algorithms
 */
public interface Classifier {
	
	public void train(List<Example> examples);
	
	public boolean predict(boolean[] example);
}

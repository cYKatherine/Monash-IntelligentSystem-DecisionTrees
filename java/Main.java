import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;


public class Main {
	/**
	 * Read in a data set in CSV format
	 * @param filename the file to read from
	 * @return the data set
	 */
	public static List<Example> readData(String filename) {
		try {
			BufferedReader br = new BufferedReader(new FileReader(filename));
			String line = null;
			ArrayList<Example> data = new ArrayList<Example>();
			while ((line = br.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, ", ");
				ArrayList<Boolean> entries = new ArrayList<Boolean>();
				if (!st.hasMoreTokens()) {
					continue;
				}
				while (st.hasMoreTokens()) {
					String s =  st.nextToken();
					if (s.equals("1") || s.equals("True") || s.equals("true")) {
						entries.add(true);
					} else if (s.equals("0") || s.equals("False") || s.equals("false")) {
						entries.add(false);
					} else {
						System.out.println("Error: expected true/false 1/0, got " + s);
						return null;
					}
				}
				if (data.size() != 0 && (
					data.get(data.size() - 1).getData().length != entries.size() - 1)) {
					System.out.println("Error: expected " + 
							data.get(data.size() - 1).getData().length + 
							" features got " + (entries.size() - 1));
					return null;
				}
				if (entries.size() <= 1) {
					System.out.println("Error: expected 2 or more entries, got " +
							entries.size());
				}
				boolean label = entries.get(entries.size() - 1);
				boolean[] vector = new boolean[entries.size() - 1];
				for (int i = 0; i < entries.size() - 1; i++) {
					vector[i] = entries.get(i);
				}
				Example example = new Example(vector, label);
				data.add(example);
			}
			br.close();
			return data;
		} catch (FileNotFoundException e) {
			System.out.println(e);
			return null;
		} catch (IOException e) {
			System.out.println(e);
			return null;
		}
	}
	
	public static double computeAccuracy(Classifier c, List<Example> testData) {
		int numRight = 0;
		for (int i = 0; i < testData.size(); i++) {
			if (testData.get(i).getLabel() == c.predict(testData.get(i).getData())) {
				numRight++;
			}
		}
		return numRight * 1.0 / testData.size();
	}
	
	public static void main(String[] args) {
	}
		
}

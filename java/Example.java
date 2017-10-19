
/**
 * A single training example.  Contains a data vector (assumed to be binary)
 * and a class label (also assumed to be binary)
 */
public class Example {
	private boolean[] data;
	private boolean label;
	
	public Example(boolean[] data, boolean label) {
		this.data = data;
		this.label = label;
	}
	
	public boolean[] getData() {
		return data;
	}
	public boolean getLabel() {
		return label;
	}
	
	public boolean equals(Object o) {
		if (o instanceof Example) {
			Example e = (Example) o;
			boolean eq = e.label == label;
			for (int i = 0; i < data.length && eq; i++) {
				eq = eq && (e.data[i] == data[i]);
			}
			return eq;
		} else {
			return false;
		}
	}
	
	public int hashCode() {
		int result = label ? 100 : 200;
		for (int i = 0; i < data.length; i++) {
			if (data[i]) {
				result += i * 300;
			}
		}
		return result;
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < data.length; i++) {
			sb.append(data[i] ? "1 " : "0 ");
		}
		sb.append(": ");
		sb.append(label);
		return sb.toString();
	}
}

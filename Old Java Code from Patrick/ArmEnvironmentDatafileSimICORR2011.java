package myoelectricarm.environments;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Random;

import johnny5.Simulator;
import johnny5.gui.GUIFactory;
import johnny5.render.RotationInfo;
import rltoys.algorithms.representations.actions.Action;
import rltoys.environments.envio.actions.ActionArray;
import rltoys.environments.envio.observations.Legend;
import rltoys.environments.envio.observations.TRStep;
import zephyr.plugin.core.api.monitoring.annotations.Monitor;

public class ArmEnvironmentDatafileSimICORR2011 {

	protected static final int CCW = 0;
	protected static final int CW = 1023;
	protected static final int ELBOW = 1;
	protected static final int WRIST = 2;
	protected static final String SENSOR1 = "ForearmAnteriorSensor";
	protected static final String SENSOR2 = "ForearmPosteriorSensor";
	protected static final String ANGLE1 = "JointAngle";
	protected static final Legend legend = new Legend(SENSOR1, SENSOR2, ANGLE1);

	// Program basics
	public Random random = new Random(0);
	private BufferedReader sourceFileReader = null;
	private String sourceFileName = null;
	private InputStream sourceFileStream = null;
	private boolean endOfExperiment = false;

	// Main Loop Setup
	private final int cycleLength = 2000;
	public double sigmin = 0;
	public double sigmax = 5.0;
	public double velmin = -1023.0;
	public double velmax = 1023.0;
	public double vScale = 1;// 10;
	public int tstep_length_ms = 10;

	// Simulator Setup
	public Simulator armSim;
	@Monitor
	private RotationInfo nextR = new RotationInfo(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

	// Averaging Constants
	public double gamma = 0.05; // 0.05
	private final double gammaR = 1 / (double) cycleLength;
	private final double gammaRU = 0.01;

	// Logged Data Fields
	@Monitor
	private double sensor1_tp1;
	@Monitor
	private double sensor1_avg;
	@Monitor
	private double sensor1_avg_norm;
	@Monitor
	private double sensor2_tp1;
	@Monitor
	private double sensor2_avg;
	@Monitor
	private double sensor2_avg_norm;
	@Monitor
	private double sensor3_tp1;
	@Monitor
	private double sensor3_avg;
	@Monitor
	private double sensor3_avg_norm;
	@Monitor
	private double sensor4_tp1;
	@Monitor
	private double sensor4_avg;
	@Monitor
	private double sensor4_avg_norm;
	@Monitor
	private double sensorR_tp1;
	@Monitor
	private double sensorR_avg;
	@Monitor
	private double velocity1_tp1;
	@Monitor
	private double velocity2_tp1;
	@Monitor
	private double angle1_tp1;
	@Monitor
	private double angle1_tp1_norm;
	@Monitor
	private double angle2_tp1;
	@Monitor
	private double angle2_tp1_norm;
	@Monitor
	private double r_avg;
	private double ru_avg;

	private TRStep lastTRStep;

	private long time;
	@SuppressWarnings("unused")
	private double cycleTime;

	// Unused or Monitor Variables
	@SuppressWarnings("unused")
	@Monitor
	private final double linezero = 0.0;
	@SuppressWarnings("unused")
	@Monitor
	private final double linehigh = 0.5;
	@SuppressWarnings("unused")
	@Monitor
	private final double linelow = -0.5;
	@Monitor
	private double lastReward;
	@Monitor
	private double angle1_target;
	@Monitor
	private double angle2_target;
	private double angleTargetSize;
	@Monitor
	private double angle1_target_upper = angle1_target + angleTargetSize;
	@Monitor
	private double angle1_target_lower = angle1_target - angleTargetSize;
	@Monitor
	private double realtime;
	private double targetScaling;
	private TRStep lastTRStep2;
	@Monitor
	private double angle2_target_lower;
	@Monitor
	private double angle2_target_upper;
	private final boolean falseSignal;
	private int targetState = 0;

	@SuppressWarnings("unused")
	private double[] listS0 = { 0.0, 15.0, 30.0 };
	@SuppressWarnings("unused")
	private double[] listS1 = { 4.0 };
	@SuppressWarnings("unused")
	private double[] listS2 = { 18.5 };

	// @Monitor
	// private final ArmSimulatorLauncher sim;

	// ------ Constructors ------

	public ArmEnvironmentDatafileSimICORR2011(Random random, String fileName) {
		this.random = random;
		this.sourceFileName = fileName;
		initialize();

		System.out.println("Launching Arm Simulator...");
		armSim = new Simulator();
		armSim.main(null);

		GUIFactory.sim.controller.setKinValue(ELBOW + 1, (float) 1.0);
		GUIFactory.sim.controller.setKinValue(WRIST + 1, (float) -1.57);

		falseSignal = false;

	}

	// --------- Main Environment Components ------

	private double get_reward() {
		double r = -0.5;

		// Check target state conditions
		if ((int) (realtime - 10) % 40 == 0 && realtime < 300) {
			targetState = 1;
			if (sourceFileName == "./sourcedata/test7_130810_ee.txt")
				targetState = 3;
		} else if ((int) (realtime - 30) % 40 == 0 && realtime < 300) {
			targetState = 2;
			if (sourceFileName == "./sourcedata/test7_130810_ee.txt")
				targetState = 4;
		} else if ((int) realtime % 20 == 0 || realtime >= 300)
			targetState = 0;

		double tgt = 0.1;

		if (targetState == 0) {
			angle1_target = -1.57;
			angle2_target = 1.1;
			angleTargetSize = tgt;
			GUIFactory.sim.controller.targetState = 0;
			GUIFactory.sim.controller.targetAngle1 = angle1_target;
			GUIFactory.sim.controller.targetAngle2 = angle2_target;
			if (falseSignal == true)
				setLevels(new double[] { 0.0, 0.0, 0.0, 0.0 });
		} else if (targetState == 1) {
			angle1_target = -2.87;
			angle2_target = 0.5;
			angleTargetSize = tgt;
			GUIFactory.sim.controller.targetState = 1;
			GUIFactory.sim.controller.targetAngle1 = angle1_target;
			GUIFactory.sim.controller.targetAngle2 = angle2_target;
			if (falseSignal == true)
				setLevels(new double[] { 2.0, 1.0, 2.0, 1.0 });
		} else if (targetState == 2) {
			angle1_target = -0.27;
			angle2_target = 1.6;
			angleTargetSize = tgt;
			GUIFactory.sim.controller.targetState = 2;
			GUIFactory.sim.controller.targetAngle1 = angle1_target;
			GUIFactory.sim.controller.targetAngle2 = angle2_target;
			if (falseSignal == true)
				setLevels(new double[] { 1.0, 2.0, 1.0, 2.0 });
		} else if (targetState == 3) {
			angle1_target = -0.27;
			angle2_target = 0.5;
			angleTargetSize = tgt;
			GUIFactory.sim.controller.targetState = 1;
			GUIFactory.sim.controller.targetAngle1 = angle1_target;
			GUIFactory.sim.controller.targetAngle2 = angle2_target;
			if (falseSignal == true)
				setLevels(new double[] { 2.0, 1.0, 2.0, 1.0 });
		} else if (targetState == 4) {
			angle1_target = -2.87;
			angle2_target = 1.6;
			angleTargetSize = tgt;
			GUIFactory.sim.controller.targetState = 2;
			GUIFactory.sim.controller.targetAngle1 = angle1_target;
			GUIFactory.sim.controller.targetAngle2 = angle2_target;
			if (falseSignal == true)
				setLevels(new double[] { 1.0, 2.0, 1.0, 2.0 });
		}

		// Make target region bounds
		angle1_target_upper = angle1_target + angleTargetSize;
		angle2_target_upper = angle2_target + angleTargetSize;
		angle1_target_lower = angle1_target - angleTargetSize;
		angle2_target_lower = angle2_target - angleTargetSize;

		// Reward based on target angle

		// if (angle1_tp1 > angle1_target_lower && angle1_tp1 < angle1_target_upper)
		// r += 0.25;
		// if (angle2_tp1 > angle2_target_lower && angle2_tp1 < angle2_target_upper)
		// r += 0.25;
		if (angle2_tp1 > angle2_target_lower && angle2_tp1 < angle2_target_upper && angle1_tp1 > angle1_target_lower && angle1_tp1 < angle1_target_upper)
			r += 1.5;

		// Negative reward for jittering via high velocity
		// r -= 0.001 * Math.abs(velocity1_tp1);
		// r -= 0.001 * Math.abs(velocity2_tp1);

		// Reward based on raw distance to setpoint
		// r -= normalize_signal(Math.abs(angle1_tp1 - angle1_target), 0.0, 3.14);
		// r -= normalize_signal(Math.abs(angle2_tp1 - angle2_target), 0.0, 3.14);

		// Send base reward to visualizer
		setGUIBaseReward(r);

		// Don't learn if outside testing window
		// if (cycleTime > pulse2End)
		// r = 0;

		// Find out if the user had anything to say...
		double ru = getRewardBuffer(); // * 0.1
		// Calculate Average User Reward
		ru_avg = ru_avg * (1 - gammaRU) + ru;
		setGUIUserReward(ru_avg);
		r += ru_avg;

		// Calculate Average Reward
		r_avg = r_avg * (1 - gammaR) + r * gammaR;

		// For human training
		// Send current reward avg to visualizer
		// setGUITotalReward(r_avg);
		// return r_avg

		// For other training
		setGUITotalReward(r);
		return r;
	}

	private double[] update_sensor_readings() {

		// Sampling Rate
		for (int loop = 0; loop < tstep_length_ms; loop++) {

			@SuppressWarnings("unused")
			boolean nextCycle = false;

			try {
				String line = sourceFileReader.readLine();
				if (line == null) {
					// endOfExperiment = true;
					// return null;
					closeSourceFile();
					loadSourceFile();
					line = sourceFileReader.readLine();
					nextCycle = true;
				}
				line = line.replaceFirst("  ", "");
				line = line.replace("  ", " ");
				String[] readings = line.split(" ");
				sensor1_tp1 = Double.parseDouble(readings[1]);
				sensor2_tp1 = Double.parseDouble(readings[2]);
				sensor3_tp1 = Double.parseDouble(readings[3]);
				sensor4_tp1 = Double.parseDouble(readings[4]);
				realtime = Double.parseDouble(readings[0]);
				// System.out.println(line);
			} catch (IOException e) {
				e.printStackTrace();
			}

			// Complexity Increases as time increases
			// if (time > 5000000) {
			// if (nextCycle == true) {
			// targetScaling = 0.5;// random.nextDouble() / 2.0 + 0.5;
			// }
			// sensor1_tp1 *= (1 + targetScaling);
			// sensor2_tp1 *= (1 - targetScaling / 2.0);
			// }

			// Small random variations in signal magnitude
			targetScaling = (random.nextDouble() - 0.5) / 10.0;
			sensor1_tp1 *= (1 + targetScaling);
			targetScaling = (random.nextDouble() - 0.5) / 10.0;
			sensor2_tp1 *= (1 + targetScaling);
			targetScaling = (random.nextDouble() - 0.5) / 10.0;
			sensor3_tp1 *= (1 + targetScaling);
			targetScaling = (random.nextDouble() - 0.5) / 10.0;
			sensor4_tp1 *= (1 + targetScaling);

			// Make sure we don't exceed bounds of signal detection
			if (sensor1_tp1 > sigmax)
				sensor1_tp1 = sigmax;
			if (sensor2_tp1 > sigmax)
				sensor2_tp1 = sigmax;
			if (sensor3_tp1 > sigmax)
				sensor3_tp1 = sigmax;
			if (sensor4_tp1 > sigmax)
				sensor4_tp1 = sigmax;
			if (sensor1_tp1 > sigmax)
				sensor1_tp1 = sigmax;

			if (sensor1_tp1 < -sigmax)
				sensor1_tp1 = -sigmax;
			if (sensor2_tp1 < -sigmax)
				sensor2_tp1 = -sigmax;
			if (sensor3_tp1 < -sigmax)
				sensor3_tp1 = -sigmax;
			if (sensor4_tp1 < -sigmax)
				sensor4_tp1 = -sigmax;

			// Calculate REFERENCE signal (bone)
			sensorR_tp1 = 0.0;

			// Process sensor signals:
			// Compute running average, then subtract reference signal
			sensor1_avg = (1 - gamma) * sensor1_avg + gamma * (Math.abs(sensor1_tp1) - sensorR_avg);
			sensor2_avg = (1 - gamma) * sensor2_avg + gamma * (Math.abs(sensor2_tp1) - sensorR_avg);
			sensor3_avg = (1 - gamma) * sensor3_avg + gamma * (Math.abs(sensor3_tp1) - sensorR_avg);
			sensor4_avg = (1 - gamma) * sensor4_avg + gamma * (Math.abs(sensor4_tp1) - sensorR_avg);
			sensorR_avg = (1 - gamma) * sensorR_avg + gamma * Math.abs(sensorR_tp1);

		}

		return new double[] { sensor1_tp1, sensor2_avg, sensor3_tp1, sensor4_avg };

	}

	private void setLevels(double[] l) {
		sensor1_avg = l[0];
		sensor2_avg = l[1];
		sensor3_avg = l[2];
		sensor4_avg = l[3];
	}

	private void update_angle() {
		double vW = boundarycheck(velocity1_tp1 * vScale, velmax);
		double vE = boundarycheck(velocity2_tp1 * vScale, velmax);
		int[] rArr = { 0, (int) vE, (int) vW, 0, 0 };
		setJointArray(rArr, time * tstep_length_ms); //
		// GUIFactory.sim.controller.setKinValue(ELBOW + 1, (float) angle2_target);
		// GUIFactory.sim.controller.setKinValue(WRIST + 1, (float) angle1_target);
		// Check elbow/wrist constraint
		if (getJointAngle(ELBOW) < 0)
			GUIFactory.sim.controller.setKinValue(ELBOW + 1, 0);
		if (getJointAngle(WRIST) > 0.2)
			GUIFactory.sim.controller.setKinValue(WRIST + 1, (float) 0.2);
		angle1_tp1 = getJointAngle(WRIST);
		angle2_tp1 = getJointAngle(ELBOW);

	}

	// ------ Helper Methods -----

	private double boundarycheck(double g, double gmax) {
		g = g > gmax ? gmax : g;
		g = g < -gmax ? -gmax : g;
		return g;
	}

	private double normalize_signal(double sensor, double sigmin, double sigmax) {
		if (sensor < sigmin)
			sensor = sigmin;
		if (sensor > sigmax)
			sensor = sigmax;
		double norm = (sensor - sigmin) / (sigmax - sigmin);
		return norm;
	}

	public void loadNewSourceFile(String fileName) {
		this.sourceFileName = fileName;
		closeSourceFile();
		loadSourceFile();
		GUIFactory.sim.controller.dataSource = sourceFileName;
	}

	private boolean loadSourceFile() {

		try {
			FileInputStream fstream = new FileInputStream(sourceFileName);
			sourceFileStream = new DataInputStream(fstream);
			sourceFileReader = new BufferedReader(new InputStreamReader(sourceFileStream));
			// sourceFileReader = new BufferedReader(new FileReader(sourceFileName));

		} catch (IOException e) {
			System.err.println("Unable to read from file");
			return false;
		}

		return true;
	}

	public void closeSourceFile() {

		if (sourceFileStream == null)
			return;
		try {
			sourceFileStream.close();
		} catch (IOException e) {
			System.err.println("Unable to close source file");
			e.printStackTrace();
		}
	}

	// ------ Simulator Interface Methods -----

	public void setJointRotation(int rotE, int index, long tstep) {
		nextR = new RotationInfo(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tstep);
		nextR.v[index] = rotE;
		nextR.d[index] = CW;
		GUIFactory.sim.recordingStream.setNextUpdate(nextR);

	}

	public void setJointArray(int[] rot, long tstep) {
		nextR = new RotationInfo(0, 0, 0, 0, 0, CW, CW, CW, CW, CW, tstep);
		nextR.v = rot;
		// nextR.d[index] = CW;
		// GUIFactory.sim.recordingStream.setNextUpdate(nextR);
		GUIFactory.sim.controller.updateControllerDirect(nextR);

	}

	public void setSensorDisplayVector(double[] disp) {
		GUIFactory.sim.controller.updateSensorDisplayVector(disp);
	}

	public void setGUIBaseReward(double r) {
		GUIFactory.sim.controller.rewardBase = r;
	}

	public void setGUIUserReward(double r) {
		GUIFactory.sim.controller.rewardUser = r;
	}

	public void setGUITotalReward(double r) {
		GUIFactory.sim.controller.rewardTotal = r;
	}

	public double getJointAngle(int index) {
		return GUIFactory.sim.controller.getKinValue(index + 1);
	}

	public double getRewardBuffer() {
		return GUIFactory.sim.controller.getRewardBuffer(true);
	}

	// ---- Update Method -----

	private void update(ActionArray action) {
		time += 1;
		cycleTime = realtime;
		velocity1_tp1 = action.actions[0];
		velocity2_tp1 = 0;// -action.actions[0];
		update_sensor_readings();
		update_angle();
		lastReward = get_reward();
	}

	private void update(ActionArray action, ActionArray action2) {
		time += 1;
		cycleTime = realtime;
		velocity1_tp1 = action.actions[0];
		velocity2_tp1 = action2.actions[0];
		update_sensor_readings();
		update_angle();
		lastReward = get_reward();
	}

	// ------- Overrides -----

	public TRStep initialize() {

		if (sourceFileName == "./sourcedata/test2_180310.txt") {
			listS0 = new double[] { 0.0, 10.45, 20.3, 31.25, 40.5 };
			listS1 = new double[] { 4.75, 25.25 };
			listS2 = new double[] { 15.3, 35.2 };

		}

		if (sourceFileName == "./sourcedata/test1_180310.txt") {
			listS0 = new double[] { 0.0, 15.0, 30.0 };
			listS1 = new double[] { 4.0 };
			listS2 = new double[] { 18.5 };
		}

		if (sourceFileName == "./sourcedata/test4_070510.txt") {
			listS0 = new double[] { 0, 20.25, 40.61, 59.86, 79.61, 99.36, 120.36, 139.86, 160.11, 180.36, 200.36, 220.61, 239.86, 260.61, 279.86, 299.36, 320.86, 339.86, 360.11, 379.11, 399.61, 420.11, 439.86, 460.11, 479.36, 500.11, 520.36, 540.11,
					559.86, 580.11, 600.11, 620.36, 639.86 };
			listS1 = new double[] { 9.25, 49.61, 89.61, 130.11, 169.61, 209.36, 249.61, 289.86, 329.61, 369.36, 409.61, 449.36, 489.61, 529.61, 569.86, 609.31 };
			listS2 = new double[] { 29.86, 70.11, 109.86, 149.86, 190.11, 230.61, 269.86, 310.11, 349.61, 389.61, 430.11, 469.61, 510.11, 549.86, 589.61, 629.68 };

		}

		if (sourceFileName == "./sourcedata/test8_130810_ef.txt") {
			listS0 = new double[] { 0.0, 15.0, 30.0 };
			listS1 = new double[] { 4.0 };
			listS2 = new double[] { 18.5 };
		}

		if (sourceFileName == "./sourcedata/test7_130810_ee.txt") {
			listS0 = new double[] { 0.0, 15.0, 30.0 };
			listS1 = new double[] { 4.0 };
			listS2 = new double[] { 18.5 };
		}

		GUIFactory.sim.controller.dataSource = sourceFileName;
		endOfExperiment = false;
		time = 0;
		cycleTime = 0;
		realtime = 0.0;
		sensor1_avg = 0;
		sensor2_avg = 0;
		sensor1_tp1 = 0;
		sensor2_tp1 = 0;
		sensor3_avg = 0;
		sensor4_avg = 0;
		sensor3_tp1 = 0;
		sensor4_tp1 = 0;
		sensorR_avg = 0;
		sensorR_tp1 = 0;
		angle1_tp1_norm = 0;
		angle2_tp1_norm = 0;
		sensor1_avg_norm = 0;
		sensor2_avg_norm = 0;
		sensor3_avg_norm = 0;
		sensor4_avg_norm = 0;
		velocity1_tp1 = 0;
		velocity2_tp1 = 0;
		angle1_tp1 = 0;
		angle2_tp1 = 0;
		lastTRStep = new TRStep(new double[] { sensor1_avg_norm, sensor2_avg_norm, angle1_tp1_norm, angle2_tp1_norm }, lastReward);
		lastTRStep2 = new TRStep(new double[] { sensor3_avg_norm, sensor4_avg_norm, angle1_tp1_norm, angle2_tp1_norm }, lastReward);
		// anglemax = 180;
		angle1_target = 2.0;
		angleTargetSize = 0.3;
		targetScaling = 1.0;
		GUIFactory.sim.controller.rewardScale = 0.5; // Ammount of each human increment

		closeSourceFile();
		loadSourceFile();

		return lastTRStep;
	}

	public TRStep step(Action action) {
		if (action != null)
			update((ActionArray) action);

		// Normalize Values
		angle1_tp1_norm = normalize_signal(angle1_tp1, -3.14, 0.5);
		sensor1_avg_norm = normalize_signal(sensor1_avg, 0, 5.0);
		sensor2_avg_norm = normalize_signal(sensor2_avg, 0, 5.0);
		sensor3_avg_norm = normalize_signal(sensor3_avg, 0, 5.0);
		sensor4_avg_norm = normalize_signal(sensor4_avg, 0, 5.0);

		setSensorDisplayVector(new double[] { sensor1_avg_norm, sensor2_avg_norm, sensor3_avg_norm, sensor4_avg_norm });

		TRStep step_t = new TRStep(lastTRStep, action, new double[] { sensor1_avg_norm, sensor2_avg_norm, angle1_tp1_norm }, lastReward);
		// TRStep step_t = new TRStep((int) realtime, new double[] { sensor1_avg, sensor2_avg, angle1_tp1 }, action, lastReward);
		lastTRStep = step_t;
		if (endOfExperiment)
			return null;
		return step_t;
	}

	public TRStep[] step(Action action, Action action2) {
		if (action != null && action2 != null)
			update((ActionArray) action, (ActionArray) action2);

		// Normalize Values
		angle1_tp1_norm = normalize_signal(angle1_tp1, -3.14, 3.14);
		angle2_tp1_norm = normalize_signal(angle2_tp1, -3.14, 3.14);
		sensor1_avg_norm = normalize_signal(sensor1_avg - sensor2_avg, -5, 5);
		sensor2_avg_norm = normalize_signal(sensor3_avg - sensor4_avg, -5, 5);
		sensor3_avg_norm = normalize_signal(sensor1_avg - sensor2_avg, -5, 5);
		sensor4_avg_norm = normalize_signal(sensor3_avg - sensor4_avg, -5, 5);
		setSensorDisplayVector(new double[] { sensor1_avg_norm, sensor2_avg_norm, sensor3_avg_norm, sensor4_avg_norm });

		TRStep step_t = new TRStep(lastTRStep, action, new double[] { sensor1_avg_norm, sensor2_avg_norm, angle1_tp1_norm, angle2_tp1_norm }, lastReward);
		TRStep step2_t = new TRStep(lastTRStep2, action2, new double[] { sensor1_avg_norm, sensor2_avg_norm, angle1_tp1_norm, angle2_tp1_norm }, lastReward);
		// TRStep step_t = new TRStep((int) realtime, new double[] { sensor1_avg, sensor2_avg, angle1_tp1 }, action, lastReward);
		lastTRStep = step_t;
		lastTRStep2 = step2_t;
		if (endOfExperiment)
			return null;
		return new TRStep[] { step_t, step2_t };
	}

	public Legend legend() {
		return legend;
	}

	public double[] getTargets() {

		return new double[] { angle1_target, angle2_target };
	}
}

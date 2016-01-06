package myoelectricarm.environments;

import rltoys.math.ranges.Range;

public class ArmPhysics2DSim implements ArmPhysicsGeneral {

	private double comS;
	private double comE;
	private double comW;
	private double angS;
	private double angE;
	private double angW;
	private double velS;
	private double velE;
	private double velW;
	private double angStarg;
	private double angEtarg;
	private double angWtarg;
	private double angSpred;
	private double angEpred;
	private double angWpred;
	private final Range comRange = new Range(-1.0, 1.0);
	private final Range velRange = new Range(-1.0, 1.0);
	private final Range angSRange = new Range(0, 2.8);
	private final Range angERange = new Range(-2.5, 0);
	private final Range angWRange = new Range(-1.5, 1.5);

	private final double appliedTorqueMultiplierNetwons = 0.1;
	private final double linkMassSE = 1.8;
	private final double linkMassEW = 1.1;
	private final double linkMassH = 0.4;
	private final double linkLengthSE = 1;
	private final double linkLengthEW = 1;
	private final double linkLengthH = 1;

	public ArmPhysics2DSim() {
		resetAll();
	}

	private void resetAll() {
		comS = 0;
		comE = 0;
		comW = 0;
		angS = 0;
		angE = 0;
		angW = 0;
		velS = 0;
		velE = 0;
		velW = 0;
		angStarg = 0;
		angEtarg = 0;
		angWtarg = 0;
		angSpred = 0;
		angEpred = 0;
		angWpred = 0;
	}

	@Override
	public void sendJointCommand(double shoulderCommand, double elbowCommand, double wristCommand) {
		comS = shoulderCommand * appliedTorqueMultiplierNetwons;
		comE = elbowCommand * appliedTorqueMultiplierNetwons;
		comW = wristCommand * appliedTorqueMultiplierNetwons;
	}

	@Override
	public void setJointAngles(double shoulderAngleRads, double elbowAngleRads, double wristAngleRads) {
		angS = shoulderAngleRads;
		angE = elbowAngleRads;
		angW = wristAngleRads;

	}

	@Override
	public void setJointVels(double shoulderVelRadsPerSec, double elbowVelRadsPerSec, double wristVelRadsPerSec) {
		velS = shoulderVelRadsPerSec;
		velE = elbowVelRadsPerSec;
		velW = wristVelRadsPerSec;

	}

	@Override
	public void setTargetJointAngles(double shoulderAngleRads, double elbowAngleRads, double wristAngleRads) {
		angStarg = shoulderAngleRads;
		angEtarg = elbowAngleRads;
		angWtarg = wristAngleRads;

	}

	@Override
	public void setPredictionJointAngles(double shoulderAngleRads, double elbowAngleRads, double wristAngleRads) {
		angSpred = shoulderAngleRads;
		angEpred = elbowAngleRads;
		angWpred = wristAngleRads;
	}

	@Override
	public double[] getJointAngles() {
		return new double[] { angS, angE, angW };
	}

	@Override
	public double[] getJointVels() {
		return new double[] { velS, velE, velW };
	}

	@Override
	public double[] getTargetJointAngles() {
		return new double[] { angStarg, angEtarg, angWtarg };
	}

	@Override
	public double[] getPredictionJointAngles() {
		return new double[] { angSpred, angEpred, angWpred };
	}

	@Override
	public void update() {
		updateVels();
		updateAngs();
	}

	private void updateVels() {

		// Moment of inertia for a thin rod is: I=1/3ML^2
		// Instantaneous angular acceleration is: alpha = tau/I
		// Change in angular velocity is therefore: w = w + alpha

		double uS = comRange.bound(comS) / (linkMassSE * 1 / 3 * linkLengthSE * linkLengthSE);
		double uE = comRange.bound(comE) / (linkMassEW * 1 / 3 * linkLengthEW * linkLengthEW);
		double uW = comRange.bound(comW) / (linkMassH * 1 / 3 * linkLengthH * linkLengthH);

		velS = velRange.bound(velS + uS);
		velE = velRange.bound(velE + uE);
		velW = velRange.bound(velW + uW);

	}

	private void updateAngs() {

		double uS = velS * 0.1;
		double uE = velE * 0.1;
		double uW = velW * 0.1;

		velS = angSRange.in(angS + uS) ? velS : 0;
		velE = angERange.in(angE + uE) ? velE : 0;
		velW = angWRange.in(angW + uW) ? velW : 0;

		angS = angSRange.bound(angS + uS);
		angE = angERange.bound(angE + uE);
		angW = angWRange.bound(angW + uW);

	}

	@Override
	public void setTargetJointCommand(double shoulderAngleRads, double elbowAngleRads, double wristAngleRads) {
		// TODO Auto-generated method stub

	}

}

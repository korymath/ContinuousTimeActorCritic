package myoelectricarm.environments;

public interface ArmPhysicsGeneral {

	public void sendJointCommand(double shoulderCommand, double elbowCommand, double wristCommand);

	public void setJointAngles(double shoulderAngleRads, double elbowAngleRads, double wristAngleRads);

	public void setJointVels(double shoulderVelRadsPerSec, double elbowVelRadsPerSec, double wristVelRadsPerSec);

	public void setTargetJointAngles(double shoulderAngleRads, double elbowAngleRads, double wristAngleRads);

	public void setTargetJointCommand(double shoulderAngleRads, double elbowAngleRads, double wristAngleRads);

	public void setPredictionJointAngles(double shoulderAngleRads, double elbowAngleRads, double wristAngleRads);

	public double[] getJointAngles();

	public double[] getJointVels();

	public double[] getTargetJointAngles();

	public double[] getPredictionJointAngles();

	public void update();

}

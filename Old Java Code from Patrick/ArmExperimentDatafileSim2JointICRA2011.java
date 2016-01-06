package myoelectricarm;

import java.io.IOException;
import java.text.NumberFormat;
import java.util.Random;

import johnny5.gui.GUIFactory;
import myoelectricarm.environments.ArmEnvironmentDatafileSimICRA2011;
import myoelectricarm.utils.TDLambdaAvgR;
import myoelectricarm.utils.TileCodingAgent;
import rltoys.algorithms.learning.control.actorcritic.onpolicy.Actor;
import rltoys.algorithms.learning.control.actorcritic.onpolicy.ActorCritic;
import rltoys.algorithms.learning.control.actorcritic.onpolicy.ActorLambda;
import rltoys.algorithms.learning.control.actorcritic.policystructure.NormalDistributionSkewed;
import rltoys.algorithms.learning.predictions.td.TD;
import rltoys.algorithms.representations.acting.PolicyDistribution;
import rltoys.algorithms.representations.tilescoding.TileCoders;
import rltoys.algorithms.representations.tilescoding.TileCodersNoHashing;
import rltoys.algorithms.representations.traces.ATraces;
import rltoys.environments.envio.actions.ActionArray;
import rltoys.environments.envio.observations.TRStep;
import zephyr.plugin.core.api.Zephyr;
import zephyr.plugin.core.api.monitoring.annotations.Monitor;
import zephyr.plugin.core.api.monitoring.fileloggers.FileLogger;
import zephyr.plugin.core.api.synchronization.Clock;

public class ArmExperimentDatafileSim2JointICRA2011 implements Runnable {

	// Program basics
	// private final String sourceFileName = "./sourcedata/test2_180310.txt";
	// private final String sourceFileName = "./sourcedata/test4_070510.txt";
	private final String sourceFileName = "./sourcedata/test8_130810_ef.txt";
	private final String testFileName = "./sourcedata/test12_160810_ef.txt";
	// private final String testFileName = "./sourcedata/test8_130810_ef.txt";
	// private final String sourceFileName = "./sourcedata/test7_130810_ee.txt";
	// private final String testFileName = "./sourcedata/test7_130810_ee.txt";
	private final String thisSourceFile = "./src/myoelectricarm/ArmExperimentACnormDatafileSim2Joint.java";
	// private final String agentFileName = "./sourcedata/agent1-21Mht.save";
	// private final String agentFileName2 = "./sourcedata/agent2-21Mht.save";

	private final Clock clock;
	public FileLogger logger;
	public Random random = new Random(667);
	public double sigmin = 0.0;
	public double sigmax = 1.0;

	// Learning Variables
	private final double alpha_c = 0.1;
	private final double alpha_a1 = 0.01;
	private final double lambda_c = 0.3;
	private final double lambda_a = 0.3;
	private final double xi = 0.0;
	private final double gamma = 0.99;
	private final double sigma_c = 1023;
	// SigmaMax = none ; SigmaMin = 1

	// Data Fields
	private final TileCodingAgent agent;
	private final TileCodingAgent agent2;
	private ActionArray a_tp1;
	private ActionArray a2_tp1;
	@Monitor
	private final ArmEnvironmentDatafileSimICRA2011 environment;

	// Unused or Monitor Variables
	@SuppressWarnings("unused")
	@Monitor
	private final double linezero = 0.0;
	@SuppressWarnings("unused")
	@Monitor
	private final double linehigh = 1;
	@SuppressWarnings("unused")
	@Monitor
	private final double linelow = -1;
	private TRStep step_t;
	@Monitor
	private TRStep[] steps;
	private TRStep step2_t;
	private final NumberFormat format = NumberFormat.getNumberInstance();

	// private final ArmSimulatorLauncher sim;

	public ArmExperimentDatafileSim2JointICRA2011() {
		// Set up the logging system
		clock = new Clock();

		// Make the learning agent
		agent = createAgentACnorm();
		agent2 = createAgentACnorm();
		// agent = (Agent) Utils.load(agentFileName);
		// agent2 = (Agent) Utils.load(agentFileName2);

		environment = new ArmEnvironmentDatafileSimICRA2011(random, sourceFileName);
		// sim = new ArmSimulatorLauncher();

	}

	@Override
	public void run() {

		// System.out.println("Launching Simulator Thread...");
		// Thread simThread = new Thread(sim, "Arm-Simulator");
		// simThread.start();

		System.out.println("Initializing experiment...");

		// Open up a log file on the disk
		try {
			logger = new FileLogger("./results/arm_results_full.txt");
		} catch (IOException e) {
			e.printStackTrace();
		}
		logger.add(this);

		// Make sure Zephyr knows where to look for logged fields
		Zephyr.advertise(clock, this);

		logger.update(0);

		step_t = environment.initialize();
		step2_t = new TRStep(step_t, step_t.r_tp1);

		// a_tp1 = (ActionArray) agent.getAtp1(step_t);
		// a2_tp1 = (ActionArray) agent2.getAtp1(step2_t);

		double targ = 0;
		double scale = 2;// 2
		int testStart = 300001;
		int testLength = 100000;

		System.out.println("Beginning main iteration...");
		// Main Loop Start
		while (true) {

			clock.tick();

			if (step_t.time == testStart + testLength) {
				environment.loadNewSourceFile(testFileName);
			}

			if (step_t.time == testStart + 2 * testLength) {
				break;
			}

			// Based on observation of s_tp1 and r_t, choose a_tp1
			// This also applies learning rules to agent
			if (step_t.time > 50000 * scale && step_t.time < testStart) {
				a_tp1 = (ActionArray) agent.getAtp1(step_t);
				GUIFactory.sim.controller.wristLearn = true;
			} else if (step_t.time >= testStart) {
				a_tp1 = (ActionArray) agent.proposeAction(step_t);
				GUIFactory.sim.controller.wristLearn = false;
			} else {
				agent.updateCritic(step_t);
				a_tp1 = new ActionArray(0.0);
				targ = environment.getTargets()[0];
				GUIFactory.sim.controller.setKinValue(3, (float) targ);
				GUIFactory.sim.controller.wristLearn = false;
			}

			// Staged task... add second agent at a later time
			if ((step2_t.time <= 50000 * scale || step2_t.time >= 100000 * scale) && step_t.time < testStart) {
				a2_tp1 = (ActionArray) agent2.getAtp1(step2_t);
				GUIFactory.sim.controller.elbowLearn = true;
			} else if (step2_t.time >= testStart) {
				a2_tp1 = (ActionArray) agent2.proposeAction(step2_t);
				GUIFactory.sim.controller.elbowLearn = false;
			} else {
				agent2.updateCritic(step2_t);
				a2_tp1 = new ActionArray(0.0);
				targ = environment.getTargets()[1];
				GUIFactory.sim.controller.setKinValue(2, (float) targ);
				GUIFactory.sim.controller.elbowLearn = false;
			}

			steps = environment.step(a_tp1, a2_tp1);
			step_t = steps[0];
			step2_t = steps[1];

			// Save Agents
			// if ((step_t.timeStep % 100000 == 0 || step_t.timeStep == 1) && step_t.timeStep < testStart) {
			// String tString1 = "results/agent1-ts" + format.format(step_t.timeStep) + ".save";
			// String tString2 = "results/agent2-ts" + format.format(step_t.timeStep) + ".save";
			// Utils.save(agent, tString1);
			// Utils.save(agent2, tString2);
			// }

			if (step_t == null)
				break;

			if (logger != null)
				logger.update(step_t.time);

			// try {
			// Thread.sleep((long) 0.5);
			// } catch (InterruptedException e) {
			// // TODO Auto-generated catch block
			// e.printStackTrace();
			// }

		}
		// System.out.println(new File(logger.filepath).getAbsolutePath());
		logger.close();
		environment.closeSourceFile();

		try {
			Runtime rt = Runtime.getRuntime();
			rt.exec("cp " + thisSourceFile + " ./results/");
			Runtime rt2 = Runtime.getRuntime();
			rt2.exec("./processDataICRA2011.sh");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("ERROR: output processing failed.");
			e.printStackTrace();
		}
	}

	@SuppressWarnings("unused")
	private TileCodingAgent createAgentACnormNoLambda() {
		// TileCoders tilesCoder = new TileCoders(4, sigmin, sigmax);
		// tilesCoder.addFullTilings(6, 6);
		// NormalMeanVarianceEligibilityActorMod actor = new NormalMeanVarianceEligibilityActorMod(stdev_a1, lambda, random, alpha_a1 / tilesCoder.nbActive(), tilesCoder.vectorSize());
		// Critic critic = new CriticEligibility(xi, alpha_c / tilesCoder.nbActive(), lambda, tilesCoder.vectorSize());
		// Control control = new ActorCriticControl(critic, actor);
		// System.out.println("Feature Vector Size:" + format.format(tilesCoder.vectorSize()));
		// System.out.println("Number of Active Features:" + format.format(tilesCoder.nbActive()));
		// return new TileCodingAgent(control, tilesCoder);

		TileCoders tileCoders = new TileCodersNoHashing(4, sigmin, sigmax);
		tileCoders.addFullTilings(5, 5);
		tileCoders.addFullTilings(8, 8);
		tileCoders.addFullTilings(12, 12);
		tileCoders.addFullTilings(20, 20);
		// tileCoders.addIndependentTilings(10, 10);
		tileCoders.includeActiveFeature();
		int vectorSize = tileCoders.vectorSize();
		int nbActive = tileCoders.nbActive();
		TD critic = new TD(gamma, alpha_c / nbActive, vectorSize);
		PolicyDistribution policyDistribution = new NormalDistributionSkewed(random, 0.0, 100.0);
		Actor actor = new Actor(policyDistribution, alpha_a1 / nbActive, vectorSize);
		ActorCritic actorCritic = new ActorCritic(critic, actor);
		return new TileCodingAgent(tileCoders, actorCritic);

	}

	private TileCodingAgent createAgentACnorm() {
		TileCoders tileCoders = new TileCodersNoHashing(4, sigmin, sigmax);
		tileCoders.addFullTilings(5, 25);
		tileCoders.addFullTilings(8, 25);
		tileCoders.addFullTilings(12, 25);
		tileCoders.addFullTilings(20, 25);
		// tileCoders.addIndependentTilings(10, 10);
		// tileCoders.addIndependentTilings(5, 5);
		tileCoders.includeActiveFeature();
		int vectorSize = tileCoders.vectorSize();
		int nbActive = tileCoders.nbActive();
		TD critic = new TDLambdaAvgR(lambda_c, xi, gamma, alpha_c / nbActive, vectorSize, new ATraces());
		PolicyDistribution policyDistribution = new NormalDistributionSkewed(random, 0.0, sigma_c);
		// Zephyr.advertize(clock, policyDistribution);
		Actor actor = new ActorLambda(lambda_a, policyDistribution, alpha_a1 / nbActive, vectorSize, new ATraces());
		ActorCritic actorCritic = new ActorCritic(critic, actor);

		System.out.println("Feature Vector Size:" + format.format(vectorSize));
		System.out.println("Number of Active Features:" + format.format(nbActive));

		return new TileCodingAgent(tileCoders, actorCritic);

	}

}

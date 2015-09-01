package org.testMavenizedJCuda;

/*
 * JCudpp - Java bindings for CUDPP, the CUDA Data Parallel
 * Primitives Library, to be used with JCuda<br />
 * http://www.jcuda.org
 *
 * Copyright 2009-2012 Marco Hutter - http://www.jcuda.org
 */
import static jcuda.jcudpp.CUDPPHashTableType.CUDPP_BASIC_HASH_TABLE;
import static jcuda.jcudpp.JCudpp.*;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaMemcpyKind.*;
import jcuda.*;
import jcuda.jcudpp.*;
import jcuda.runtime.*;

/**
 * A sample demonstrating the use of the hashing functions of CUDPP 2.0. Note
 * that this sample requires a device with a Compute Capability >= 2.0.
 */
public class JCudppHashSample {
	/**
	 * Entry point of this sample
	 * 
	 * @param args
	 *            Not used
	 */
	public static void main(String[] args) {
		// Enable exceptions and omit subsequent error checks
		JCudpp.setExceptionsEnabled(true);
		JCuda.setExceptionsEnabled(true);

		// First check if the device has a compute capability that is
		// sufficient for using the CUDPP hash table. CUDPP would
		// otherwise report an 'CUDPP_ERROR_ILLEGAL_CONFIGURATION'.
		int device[] = { -1 };
		JCuda.cudaGetDevice(device);
		cudaDeviceProp prop = new cudaDeviceProp();
		JCuda.cudaGetDeviceProperties(prop, device[0]);
		if (prop.major < 2) {
			System.err.println("CUDPP hash table requires a device with"
					+ " compute capability >= 2, but device " + device[0]
					+ " has only " + prop.major + "." + prop.minor);
			return;
		}

		// Definition of the size and space usage of the hash table
		final int size = 1000;
		final int num = 100;
		final float spaceUsage = 1.5f;

		// Create a handle for CUDPP
		CUDPPHandle theCudpp = new CUDPPHandle();
		cudppCreate(theCudpp);

		// Create a plan for a basic hash table
		CUDPPHandle plan = new CUDPPHandle();
		CUDPPHashTableConfig config = new CUDPPHashTableConfig();
		config.type = CUDPP_BASIC_HASH_TABLE;
		config.kInputSize = size;
		config.space_usage = spaceUsage;

		// Create the hash table
		cudppHashTable(theCudpp, plan, config);

		// Create the host data for the keys and values
		// that will be inserted into the hash table
		int h_inKeys[] = new int[num];
		int h_inVals[] = new int[num];
		for (int i = 0; i < num; i++) {
			h_inKeys[i] = i;
			h_inVals[i] = num - i;
		}

		// Allocate device data for the keys and values,
		// and copy the host data to the device
		Pointer d_inKeys = new Pointer();
		JCuda.cudaMalloc(d_inKeys, num * Sizeof.INT);
		cudaMemcpy(d_inKeys, Pointer.to(h_inKeys), num * Sizeof.INT,
				cudaMemcpyHostToDevice);
		Pointer d_inVals = new Pointer();
		JCuda.cudaMalloc(d_inVals, num * Sizeof.INT);
		cudaMemcpy(d_inVals, Pointer.to(h_inVals), num * Sizeof.INT,
				cudaMemcpyHostToDevice);

		// Insert the keys and values into the hash table
		cudppHashInsert(plan, d_inKeys, d_inVals, num);

		// Create the host data for the keys and values
		// that will be retrieved from the hash table
		int h_outKeys[] = new int[num];
		int h_outVals[] = new int[num];
		for (int i = 0; i < num; i++) {
			// Use a set of keys where every second
			// key is contained in the table
			if (i % 2 == 0) {
				h_outKeys[i] = i;
			} else {
				h_outKeys[i] = num + 1;
			}
		}

		// Allocate device data for the keys and values,
		// and copy the host data to the device
		Pointer d_outKeys = new Pointer();
		JCuda.cudaMalloc(d_outKeys, num * Sizeof.INT);
		cudaMemcpy(d_outKeys, Pointer.to(h_outKeys), num * Sizeof.INT,
				cudaMemcpyHostToDevice);
		Pointer d_outVals = new Pointer();
		JCuda.cudaMalloc(d_outVals, num * Sizeof.INT);

		// Retrieve the values from the hash table
		cudppHashRetrieve(plan, d_outKeys, d_outVals, num);

		// Copy the retrieved values back to the host
		cudaMemcpy(Pointer.to(h_outVals), d_outVals, num * Sizeof.INT,
				cudaMemcpyDeviceToHost);

		// Verify the result
		boolean passed = true;
		for (int i = 0; i < num; i++) {
			// Use a set of keys where every second
			// key is contained in the table
			if (i % 2 == 0) {
				if (h_outVals[i] != num - i) {
					System.err.println("For key " + h_outKeys[i] + " expected "
							+ (num - i) + " but found " + h_outVals[i]);
					passed = false;
				}
			} else {
				if (h_outVals[i] != CUDPP_HASH_KEY_NOT_FOUND) {
					System.err.println("For key " + h_outKeys[i]
							+ " expected CUDPP_HASH_KEY_NOT_FOUND"
							+ " but found " + h_outVals[i]);
					passed = false;
				}
			}
		}
		System.out.println(passed ? "PASSED" : "FAILED");

		// Clean up
		cudaFree(d_outKeys);
		cudaFree(d_outVals);
		cudaFree(d_inKeys);
		cudaFree(d_inVals);
		cudppDestroyHashTable(theCudpp, plan);
		cudppDestroy(theCudpp);
	}
}
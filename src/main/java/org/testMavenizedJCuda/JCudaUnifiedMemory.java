package org.testMavenizedJCuda;

import static jcuda.jcublas.JCublas2.*;
import static jcuda.runtime.JCuda.*;

import java.nio.*;

import jcuda.*;
import jcuda.jcublas.*;
import jcuda.runtime.JCuda;

public class JCudaUnifiedMemory
{
   public static void main(String[] args)
   {
       JCuda.setExceptionsEnabled(true);
       JCublas.setExceptionsEnabled(true);

       // Allocate managed memory that is accessible to the host
       Pointer p = new Pointer();
       int n = 10;
       long size = n * Sizeof.FLOAT;
       cudaMallocManaged(p, size, cudaMemAttachHost);
      
       
       // Obtain the byte buffer from the pointer
       ByteBuffer bb = p.getByteBuffer(0, size);
       System.out.println("Buffer on host side: "+bb);

       // Fill the buffer with sample data  
       FloatBuffer fb = bb.order(ByteOrder.nativeOrder()).asFloatBuffer();
       for (int i=0; i<n; i++)
       {
           fb.put(i, i);
       }
      
       // Make the buffer accessible to all devices
       cudaStreamAttachMemAsync(null, p, 0, cudaMemAttachGlobal);
       cudaStreamSynchronize(null);

       // Use the buffer in a device operation
       // (here, a dot product with JCublas, for example)
       cublasHandle handle = new cublasHandle();
       cublasCreate(handle);
       float result[] =  { -1.0f };
       cublasSdot(handle, n, p, 1, p, 1, Pointer.to(result));
       System.out.println(result[0]);
   }
}

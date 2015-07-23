#### TestMvnJCuda
Tests of Mavenized JCuda

Building of the project steps:

1. Clone [Mavenized JCuda](https://github.com/MysterionRise/mavenized-jcuda) and this repo.

2. Copy _mavenized-jcuda/repo/jcuda_ directory into your maven local repository.

3. After building _mavenized-jcuda_ , build this project with `mvn clean install`

4. Run java classes like JCudaDeviceQuery with:
`java -cp target/lib/\*:target/TestMvnJCuda-0.0.1-SNAPSHOT.jar org.testMavenizedJCuda.JCudaDeviceQuery`

*Steps are tested in Ubuntu x64 and Maven 3.3.3*

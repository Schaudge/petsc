
static char help[] = "Tests not trapping an underflow\n\n";

#include <petscsys.h>
#include <float.h>
#include <math.h>

/* From https://stackoverflow.com/questions/37193363/float-underflow-in-c-explanation */
void demo(void) {
  const char *format = "%.10e %a\n";
  printf(format, FLT_MIN, FLT_MIN);
  printf(format, FLT_TRUE_MIN, FLT_TRUE_MIN);

  float f = nextafterf(1.0f, 2.0f);
  do {
    f /= 2;
    printf(format, f, f);
  } while (f);
}

int main(int argc, char **argv) {
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  demo();
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -fp_trap

TEST*/

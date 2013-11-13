
!subroutine test_mkl(m, n, A, B, C)
!  use mkl95_blas, only: gemm
!  implicit none
! integer, intent(in)  :: m
!  integer, intent(in)  :: n
! double precision, intent(inout) :: A(m, n)
! double precision, intent(inout) :: B(m, n)
! double precision, intent(inout) :: C(m, n)
!  call gemm (A, B, C)
!end subroutine test_mkl

module bfconstans

double precision, parameter :: TOF_SML (3, 6) = reshape ( (/ 1,  0,  0, &
                                                            -1,  0,  0, &
                                                             0,  1,  0,  &
                                                             0, -1,  0, &
                                                             0,  0,  1,  &
                                                             0,  0, -1 /), (/3, 6/))

double precision, parameter :: PRO_OF_SML (3, 4) = reshape ( (/ 1, 0, 0, &
     -1, 0, 0,  0, 1, 0,  0, -1, 0 /), (/3, 4/))

character, parameter :: PRO_CHANS (4) = achar ((/ 49, 17, 50, 18 /))

double precision, parameter :: DEU_OF_SML (3, 4) = reshape ( (/ 1, 0, 0, &
     -1, 0, 0,  0, 0, 1,  0, 0, -1 /), (/3, 4/))
end module


subroutine bf_conv(A, X, R)
  implicit none
  double precision, intent (in) :: A(:,:), X(:,:)
  real, intent (out) :: R(:,:)
  integer :: m, n, k, c, d

  m = size (A, 1)
  n = size (A, 2)
  c = size (X, 2)
  k = size (R, 1)

  d = (m / c) * n

  R = SNGL (reshape (matmul (X, reshape(A, (/c, d/))), (/ k, n /)))

end subroutine

subroutine bf_to_lms (m, n, A, k, B, nch, C)
  use bfconstans
  implicit none
  integer, intent(in)  :: m
  integer, intent(in)  :: n
  integer, intent(in)  :: k
  integer, intent(in)  :: nch
  double precision, intent(in) :: A(m, n)
  real, intent(inout) :: B(k, n)
  character, intent (in) :: C(nch)

  interface
     subroutine bf_conv(A, X, R)
       double precision, intent (in) :: A(:,:), X(:,:)
       real, intent (out) :: R(:,:)
     end subroutine bf_conv
  end interface

  if (nch == 3) then
     B = SNGL (A)
  else if (nch == 4) then
     if (C(3) == achar (50) .and. C(4) == achar (18)) then
        call bf_conv (A, PRO_OF_SML, B)
     else if (C(3) == achar (51) .and. C(4) == achar (19)) then
        call bf_conv (A, DEU_OF_SML, B)
     else
        B(:,:) = 0
     end if
  else if (nch == 6) then
     call bf_conv (A, TOF_SML, B)
  else
     B(:,:) = 0
  end if

end subroutine

subroutine bff_normalize(m, n, A)
  implicit none
  integer, intent (in) :: m
  integer, intent (in) :: n
  real, intent (inout) :: A(m, n)
  integer :: i

  do i = 1, n
     A(:, i) = 0.5 + 0.5 * (A(:, i) / maxval (abs (A(:, i))))
  end do

end subroutine

# Copyright 2020-2021 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
using Test
using TaD
import TaD: evaluate

@testset "evaluate bspline basis functions" begin
    kv = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5.0]
    @test evaluate(BSplineBasis(kv, 2), 5 / 2)[1] ≈ 1 / 8
    @test evaluate(BSplineBasis(kv, 2), 5 / 2)[2] ≈ 3 / 4
    @test evaluate(BSplineBasis(kv, 2), 5 / 2)[3] ≈ 1 / 8
end
@testset "evaluate bspline basis function derivatives" begin
    kv = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[1, 1] ≈ 0.25
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[1, 2] ≈ 0.5
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[1, 3] ≈ 0.25
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[1, 4] ≈ 0.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[2, 1] ≈ -1.5
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[2, 2] ≈ 0.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[2, 3] ≈ 1.5
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[2, 4] ≈ 0.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[3, 1] ≈ 6.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[3, 2] ≈ -12.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[3, 3] ≈ 6.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[3, 4] ≈ 0.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[4, 1] ≈ -12.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[4, 2] ≈ 48.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[4, 3] ≈ -84.0
    @test evaluate(BSplineBasis(kv, 3, k=3), 0.5)[4, 4] ≈ 48.0
end
@testset "evaluate bspline curve and derivatives" begin
    kv = [0, 0, 0, 1, 1, 1]
    b = BSplineBasis(kv, 2, k=1)
    cp = [0 0; 0.5 1; 1 0]
    c = BSplineCurve(b, cp)
    @test evaluate(c, 0.05)[1, 1] ≈ 0.05
    @test evaluate(c, 0.05)[1, 2] ≈ 0.095
    @test evaluate(c, 0.05)[2, 1] ≈ 1.0
    @test evaluate(c, 0.05)[2, 2] ≈ 1.8
    @test evaluate(c, 0.5)[1, 1] ≈ 0.5
    @test evaluate(c, 0.5)[1, 2] ≈ 0.5
    @test evaluate(c, 0.5)[2, 1] ≈ 1.0
    @test evaluate(c, 0.5)[2, 2] ≈ 0.0
end

